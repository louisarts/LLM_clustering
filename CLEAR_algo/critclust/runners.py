"""End-to-end runs.

Each run produces one row per (benchmark, seed) recording the chosen candidate, the judge
score of every candidate, the repair outcome and the final NMI / ACC, and writes three CSVs
into results/:

    <stem>_runs.csv        one row per benchmark and seed
    <stem>_summary.csv     mean +- std over seeds, per benchmark
    <stem>_candidates.csv  how often each candidate was chosen, per benchmark

Gold labels enter in exactly two places: setting k, and the final scoring on the last line of
each seed. Selection, repair and every candidate are driven by the LLM reference labels alone.
"""
import numpy as np
import pandas as pd

import json

from . import config
from .data import load_bench, score
from .embeddings import spaces
from .generation import build_reference, judge_reference, rewrites, exemplars
from .judges import judge_A, judge_B, split_reference
from .llm import bind_llm
from .pools import pool_A, pool_B, pool_C
from .repair import boundary_repair


def _partial_path(algo, setting, aligned=False):
    return config.RESULTS / f'{config.run_name(algo, setting, aligned)}_runs.csv'


def _run_aligned(algo, setting, benches=None, seeds=None, verbose=True, resume=True):
    """The RQ1-aligned protocol (CritClust_A only): per-seed fresh 1,000-doc judge
    references, judge LLM per config.judge_llm_for, generation stages unchanged.
    Writes to the *_aligned stems; legacy results are untouched.
    """
    assert algo in ('A', 'C'), 'aligned protocol covers CritClust_A and CritClust_C'
    benches = benches or config.benchmarks_for(setting)
    seeds = seeds or config.SEEDS
    crit = config.criteria()

    rows, done = [], set()
    partial = _partial_path(algo, setting, aligned=True)
    if resume and partial.exists():
        prev = pd.read_csv(partial)
        rows = prev.to_dict('records')
        counts = prev.groupby('bench').seed.nunique()
        done = set(counts[counts >= len(seeds)].index)
        if done:
            print(f'resuming: {len(done)} benchmarks already complete, skipping them')

    failed = []
    for bench in list(benches) + ['__RETRY__']:
        if bench == '__RETRY__':
            if not failed:
                break
            print(f'retrying {len(failed)} benchmark(s) that failed setup: {failed}')
            benches_retry, failed = failed, []
            for b in benches_retry:
                _one_benchmark_aligned(b, algo, setting, seeds, crit, rows, partial,
                                       failed, verbose)
            if failed:
                print(f'  !! still failing after retry, absent from results: {failed}')
            break
        if bench in done:
            continue
        _one_benchmark_aligned(bench, algo, setting, seeds, crit, rows, partial,
                               failed, verbose)
    return pd.DataFrame(rows)


def _one_benchmark_aligned(bench, algo, setting, seeds, crit, rows, partial, failed,
                           verbose):
    try:
        docs, gold, k = load_bench(bench)
        criterion = crit[bench]
        spec = {'criterion': criterion}
        embedder, llm = config.models_for(bench, setting)
        judge_llm = config.judge_llm_for(setting)
        llm_src = config.llm_source_setting(setting)

        # generation artifacts: the setting's own LLM, all cached. CritClust_C shares
        # CritClust_A's judge instrument and generation codebook by construction.
        art = 'A' if algo == 'C' else algo
        ask_gen = bind_llm(llm)
        cb_gen = json.loads((config.CODEBOOKS /
                             f'{art}_{bench}_{llm_src}_codebook.json').read_text())
        rewrite_texts = rewrites(bench, llm_src, docs, criterion, ask_gen)
        exemplar_map = exemplars(bench, llm_src, cb_gen, criterion, ask_gen)
        Xr, Xw, Xc, P = spaces(bench, setting, docs, rewrite_texts, exemplar_map)
    except Exception as e:
        print(f'  !! {bench}: setup failed ({type(e).__name__}: {e}) - skipped')
        failed.append(bench)
        return

    bench_rows = []
    for seed in seeds:
        try:
            ask_judge = bind_llm(judge_llm)
            fm, cb_judge, cb_method = judge_reference(bench, llm_src, art, docs,
                                                      criterion, k, ask_judge, seed)
        except Exception as e:
            print(f'  !! {bench} s{seed}: judge reference failed '
                  f'({type(e).__name__}: {e}) - seed skipped')
            continue
        if len(fm) < 60:
            print(f'  {bench} s{seed}: reference too small ({len(fm)}), seed skipped')
            continue

        score_fn, select_fn = judge_A(fm, len(docs))
        ask_gen = bind_llm(llm)                    # repair runs on the generation LLM
        if algo == 'C':
            cands, cand_spaces = pool_C(Xr, Xw, Xc, P, k, seed)
        else:
            cands, cand_spaces = pool_A(Xr, Xc, P, k, seed)
        winner, judge_scores = select_fn(cands)
        labels, rounds_kept, n_calls = boundary_repair(
            docs, cands[winner], cand_spaces[winner], k, criterion, spec,
            score_fn, ask_gen, tag=f'{algo}_{setting}_aligned_{bench}_s{seed}')

        nmi_val, acc_val = score(gold, labels)
        row = {
            'algo': f'CritClust_{algo}', 'setting': setting, 'protocol': 'aligned',
            'bench': bench, 'k': k, 'seed': seed, 'winner': winner,
            'rounds_kept': rounds_kept, 'llm_calls': n_calls,
            'n_categories': len(cb_judge), 'n_ref': len(fm), 'n_judge': len(fm),
            'nmi': nmi_val, 'acc': acc_val, 'embedder': embedder, 'llm': llm,
            'judge_llm': judge_llm, 'judge_codebook': cb_method,
        }
        row.update({f'judge_{name}': round(v, 4) for name, v in judge_scores.items()})
        bench_rows.append(row)
        if verbose:
            print(f'{algo}/{setting}/aligned {bench:17} s{seed} -> {winner:17} '
                  f'nmi {nmi_val:5} acc {acc_val:5} (repair kept {rounds_kept})',
                  flush=True)

    rows.extend(bench_rows)
    pd.DataFrame(rows).to_csv(partial, index=False)


def _run(algo, setting, benches=None, seeds=None, verbose=True, resume=True):
    """Rows are checkpointed to the runs CSV after every benchmark.

    A crash therefore costs at most one benchmark, and re-invoking with resume=True skips
    benchmarks whose seeds are already complete.
    """
    benches = benches or config.benchmarks_for(setting)
    seeds = seeds or config.SEEDS
    crit = config.criteria()

    rows, done = [], set()
    partial = _partial_path(algo, setting)
    if resume and partial.exists():
        prev = pd.read_csv(partial)
        rows = prev.to_dict('records')
        counts = prev.groupby('bench').seed.nunique()
        done = set(counts[counts >= len(seeds)].index)
        if done:
            print(f'resuming: {len(done)} benchmarks already complete, skipping them')

    failed = []
    for bench in list(benches) + ['__RETRY__']:
        if bench == '__RETRY__':
            # Setup failures are usually transient - a gateway timeout, a cold model. Skipping
            # keeps one bad benchmark from killing the run, but a run that quietly ends at 18/19
            # is worse than a slow one, so everything skipped gets one more attempt here.
            if not failed:
                break
            print(f'retrying {len(failed)} benchmark(s) that failed setup: {failed}')
            benches_retry, failed = failed, []
            for b in benches_retry:
                _one_benchmark(b, algo, setting, seeds, crit, rows, partial, failed, verbose)
            if failed:
                print(f'  !! still failing after retry, absent from results: {failed}')
            break
        if bench in done:
            continue
        _one_benchmark(bench, algo, setting, seeds, crit, rows, partial, failed, verbose)

    return pd.DataFrame(rows)


def _one_benchmark(bench, algo, setting, seeds, crit, rows, partial, failed, verbose):
    """Run every seed for one benchmark and checkpoint. Appends to `failed` on a setup error."""
    if True:
        try:
            docs, gold, k = load_bench(bench)
            criterion = crit[bench]
            spec = {'criterion': criterion}
            embedder, llm = config.models_for(bench, setting)
            ask = bind_llm(llm)

            # LLM artifacts depend only on the LLM: hybrids load them from the cache of
            # the run that shares their LLM (llm_src == setting for the pure settings).
            llm_src = config.llm_source_setting(setting)
            fm, codebook = build_reference(bench, llm_src, algo, docs, criterion, k, ask)
        except Exception as e:
            print(f'  !! {bench}: setup failed ({type(e).__name__}: {e}) - skipped')
            failed.append(bench)
            return
        if len(fm) < 60:
            print(f'  {bench}: reference too small ({len(fm)} usable), skipped')
            return

        try:
            rewrite_texts = rewrites(bench, llm_src, docs, criterion, ask)
            exemplar_map = exemplars(bench, llm_src, codebook, criterion, ask)
            Xr, Xw, Xc, P = spaces(bench, setting, docs, rewrite_texts, exemplar_map)
        except Exception as e:
            print(f'  !! {bench}: embedding setup failed ({type(e).__name__}: {e}) - skipped')
            failed.append(bench)
            return

        bench_rows = []
        for seed in seeds:
            if algo == 'A':
                score_fn, select_fn = judge_A(fm, len(docs))
                cands, cand_spaces = pool_A(Xr, Xc, P, k, seed)
                n_judge = len(fm)
            else:
                train_ids, judge_ids = split_reference(fm, seed)
                score_fn, select_fn = judge_B(fm, judge_ids, seed)
                cands, cand_spaces = pool_B(Xr, Xw, Xc, P, k, seed, fm, train_ids)
                n_judge = len(judge_ids)

            winner, judge_scores = select_fn(cands)
            labels, rounds_kept, n_calls = boundary_repair(
                docs, cands[winner], cand_spaces[winner], k, criterion, spec,
                score_fn, ask, tag=f'{algo}_{setting}_{bench}_s{seed}')

            nmi_val, acc_val = score(gold, labels)
            row = {
                'algo': f'CritClust_{algo}', 'setting': setting, 'bench': bench, 'k': k,
                'seed': seed, 'winner': winner, 'rounds_kept': rounds_kept,
                'llm_calls': n_calls, 'n_categories': len(codebook), 'n_ref': len(fm),
                'n_judge': n_judge, 'nmi': nmi_val, 'acc': acc_val,
                'embedder': embedder, 'llm': llm,
            }
            row.update({f'judge_{name}': round(v, 4) for name, v in judge_scores.items()})
            bench_rows.append(row)

            if verbose:
                print(f'{algo}/{setting} {bench:17} s{seed} -> {winner:17} '
                      f'nmi {nmi_val:5} acc {acc_val:5} (repair kept {rounds_kept})',
                      flush=True)

        rows.extend(bench_rows)
        pd.DataFrame(rows).to_csv(partial, index=False)      # checkpoint


def run_A(setting, benches=None, seeds=None, verbose=True, aligned=False):
    """CritClust_A: V0, unchanged. aligned=True uses the RQ1-aligned judge protocol."""
    if aligned:
        return _run_aligned('A', setting, benches, seeds, verbose)
    return _run('A', setting, benches, seeds, verbose)


def run_C(setting, benches=None, seeds=None, verbose=True, aligned=True):
    """CritClust_C: CritClust_A with the pool extended to eight judge-independent
    candidates. Aligned protocol only."""
    if not aligned:
        raise NotImplementedError('CritClust_C exists only under the aligned protocol')
    return _run_aligned('C', setting, benches, seeds, verbose)


def run_B(setting, benches=None, seeds=None, verbose=True, aligned=False):
    """CritClust_B: V4 with an aim-for-k codebook and boundary repair restored."""
    if aligned:
        raise NotImplementedError('aligned protocol is CritClust_A only')
    return _run('B', setting, benches, seeds, verbose)


def save(df, algo, setting, aligned=False):
    """Write the three CSVs and return (summary, candidate_counts)."""
    df = df.drop_duplicates(['bench', 'seed'], keep='last').reset_index(drop=True)
    stem = config.run_name(algo, setting, aligned)
    df.to_csv(config.RESULTS / f'{stem}_runs.csv', index=False)

    summary = (df.groupby('bench')
                 .agg(k=('k', 'first'), seeds=('seed', 'count'),
                      nmi_mean=('nmi', 'mean'), nmi_std=('nmi', 'std'),
                      acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
                      rounds_kept=('rounds_kept', 'mean'),
                      n_categories=('n_categories', 'first'))
                 .round(2).reset_index())
    summary.to_csv(config.RESULTS / f'{stem}_summary.csv', index=False)

    candidates = (df.pivot_table(index='bench', columns='winner', values='seed',
                                 aggfunc='count').fillna(0).astype(int))
    candidates.to_csv(config.RESULTS / f'{stem}_candidates.csv')

    print(f'wrote {stem}_runs.csv, _summary.csv and _candidates.csv to {config.RESULTS}')
    return summary, candidates


def load(algo, setting):
    """Read back a completed run. -> (runs, summary, candidates)."""
    stem = config.run_name(algo, setting)
    runs = pd.read_csv(config.RESULTS / f'{stem}_runs.csv')
    summary = pd.read_csv(config.RESULTS / f'{stem}_summary.csv')
    candidates = pd.read_csv(config.RESULTS / f'{stem}_candidates.csv', index_col=0)
    return runs, summary, candidates


def available(algo, setting):
    """True when this run has finished and written all three CSVs."""
    stem = config.run_name(algo, setting)
    return all((config.RESULTS / f'{stem}_{part}.csv').exists()
               for part in ('runs', 'summary', 'candidates'))


def load_if_available(algo, setting):
    """-> (runs, summary, candidates) or None. Lets the notebook skip runs not yet done.

    A run that is still in progress has a checkpointed _runs.csv but no _summary.csv, so it
    is reported as unavailable rather than plotted half-finished.
    """
    if not available(algo, setting):
        return None
    return load(algo, setting)
