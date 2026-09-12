#!/usr/bin/env python
"""Ten independent judge runs with the judge LLM set to gpt-4o - the fifth judge
bed. gpt-4o is the last matched-stack LLM in Research Question 4 without a
validity bed (the k-LLMmeans record holder on Banking77 and MASSIVE(D)), and it
completes the within-family OpenAI capability curve (gpt-3.5-turbo, gpt-4o-mini,
gpt-4o) alongside the cross-family ceiling (gemini-2.5-flash, glm-5).

Protocol mirrors 09/11/12/13 run for run (same discovery/assignment sample
seeds, N_DISCOVERY=300, N_ASSIGN=1000, batched 20-doc classification, scoring
identical, instructor_gen partition bed). ALL runs build fresh codebooks; the
128k context takes the full RQ1-faithful discovery prompt everywhere; codebook
replies cap at gpt-4o's 16,384-token output limit; chunked-discovery fallback
retained as a safety net only.

Outputs:
    data/replicates_gpt4o/run{r}/{bench}_codebook.json / {bench}_reference.csv
    results/judge_replicates_instructor_gen_gpt4o.csv
    results/replicate_meta_gpt4o.csv

Budget-guarded (hard stop $999), checkpointed per batch, resumable, run-major.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

RQ1C = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RQ1C / 'scripts'))

import judge  # noqa: E402
from judge import CRITERIA, score_partition  # noqa: E402

sys.path.insert(0, str(RQ1C.parent / 'research_question_3_clean'))
from critclust.codebooks import codebook_A_chunked  # noqa: E402
from critclust import llm as critllm  # noqa: E402

GPT4O = 'gpt-4o'
critllm.CURRENT_LLM = GPT4O   # codebook_A_chunked sizes its batches off this
RUNS = [int(x) for x in os.environ.get('RUNS', ','.join(map(str, range(1, 11)))).split(',')]
ONLY = [b for b in os.environ.get('BENCH_ONLY', '').split(',') if b]
N_DISCOVERY, N_ASSIGN = 300, 1000
BATCH, DOC_CLIP, WORKERS = 20, 1000, 8
BUDGET_STOP = 999.0
INPUT_BUDGET_TOKENS = 100000         # 128k context: full 400-char clip always fits

REP = RQ1C / 'data' / 'rq1' / 'replicates_gpt4o'
OUT_SCORES = RQ1C / 'results' / 'rq1' / 'judge_replicates_instructor_gen_gpt4o.csv'
OUT_META = RQ1C / 'results' / 'rq1' / 'replicate_meta_gpt4o.csv'

CLIENT = OpenAI(base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'], timeout=240, max_retries=2)


def ask_gpt4o(prompt, max_tokens=2000, retries=4):
    for attempt in range(retries):
        try:
            r = CLIENT.chat.completions.create(
                model=GPT4O, temperature=0, max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}])
            return r.choices[0].message.content or ''
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


def _ask_codebook_gpt4o(prompt, max_tokens=16384, retries=3):
    """Replaces judge._ask_thinking: gpt-4o caps completions at 16,384 tokens.
    Returns (text, finish_reason) as build_codebook expects."""
    for attempt in range(retries):
        try:
            r = CLIENT.chat.completions.create(
                model=GPT4O, temperature=0, max_tokens=min(max_tokens, 16384),
                messages=[{'role': 'user', 'content': prompt}])
            return (r.choices[0].message.content or ''), (r.choices[0].finish_reason or '')
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


judge.MODEL = GPT4O
judge.ask_llm = ask_gpt4o
judge._ask_thinking = _ask_codebook_gpt4o


def gateway_spend():
    try:
        base = os.environ['OPENAI_BASE_URL'].rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3].rstrip('/')
        req = urllib.request.Request(
            f'{base}/key/info',
            headers={'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return float(json.load(r)['info']['spend'])
    except Exception:
        return None


def guard():
    s = gateway_spend()
    if s is not None and s > BUDGET_STOP:
        raise RuntimeError(f'BUDGET STOP at ${s:.2f}')


def _clip(doc, max_chars=DOC_CLIP):
    return ' '.join(str(doc).split())[:max_chars]


def codebook_text(codebook):
    return '\n'.join(
        f"{c['id']}. {c['name']}" + (f" - {c['definition']}" if c.get('definition') else '')
        for c in codebook)


def parse_labels(txt):
    try:
        arr = json.loads(txt[txt.find('['): txt.rfind(']') + 1])
    except Exception:
        return None
    out = {}
    for pos, item in enumerate(arr, start=1):
        # gpt-3.5 sometimes answers with category NAMES ("physics.optics") instead of
        # ids; those entries are dropped, so a mostly-name reply fails parsing and takes
        # the standard remedy (retry once, then per-document fallback).
        try:
            if isinstance(item, dict):
                doc, cat = item.get('doc'), item.get('category')
                if doc is None or cat is None:
                    continue
                out[int(doc)] = int(cat)
            elif isinstance(item, (int, float)):
                out[pos] = int(item)
        except (ValueError, TypeError):
            continue
    return out or None


def discovery_clip(texts, rng_idx):
    """400 chars per document unless that prompt overruns the context; then 150."""
    est = (800 + sum(min(len(' '.join(str(texts[i]).split())), 400) + 8
                     for i in rng_idx)) / 3.6
    return 400 if est <= INPUT_BUDGET_TOKENS else 150


def get_codebook(run, bench, texts):
    f = REP / f'run{run}' / f'{bench}_codebook.json'
    if f.exists():
        d = json.loads(f.read_text())
        return d['categories'], d.get('clip_chars', 400), d.get('source', 'single_shot')
    f.parent.mkdir(parents=True, exist_ok=True)
    guard()
    spec = CRITERIA[bench]
    clip, source = 400, 'single_shot'
    for attempt in range(3):
        rng = np.random.RandomState(1000 * run + attempt)
        idx = rng.choice(len(texts), min(N_DISCOVERY, len(texts)), replace=False)
        clip = discovery_clip(texts, idx)
        try:
            cats = judge.build_codebook([texts[i] for i in idx], spec, clip_chars=clip)
            break
        except ValueError as e:
            print(f'  run{run} {bench}: codebook attempt {attempt + 1} failed ({e})',
                  flush=True)
    else:
        # safety net from the gpt-3.5 bed, in case a reply overruns the output cap
        # (fine taxonomies). Fall back to critclust's chunked discovery, the RQ3
        # remedy for small-context models: categories accumulate over 20-document
        # batches with a saturation stop; the model still picks the granularity
        # and is never told k.
        print(f'  run{run} {bench}: falling back to chunked discovery', flush=True)
        cats = codebook_A_chunked(texts, spec['criterion'], ask_gpt4o)
        source = 'chunked_no_target'
    f.write_text(json.dumps({'bench': bench, 'run': run, 'llm': GPT4O, 'source': source,
                             'clip_chars': clip, 'criterion': spec['criterion'],
                             'categories': cats}, indent=2))
    return cats, clip, source


def get_reference(run, bench, texts, codebook):
    f = REP / f'run{run}' / f'{bench}_reference.csv'
    done = {}
    if f.exists():
        d = pd.read_csv(f)
        done = dict(zip(d.doc_idx.astype(int), d.category.astype(int)))
    rng = np.random.RandomState(2000 * run + 7)
    sample = rng.choice(len(texts), min(N_ASSIGN, len(texts)), replace=False)
    pending = [int(i) for i in sample if int(i) not in done]
    if not pending:
        return done, 0
    guard()
    spec = CRITERIA[bench]
    cbtxt = codebook_text(codebook)
    chunks = [pending[i:i + BATCH] for i in range(0, len(pending), BATCH)]

    def one_chunk(chunk):
        guard()
        body = '\n'.join(f'{n}. {_clip(texts[i])}' for n, i in enumerate(chunk, 1))
        prompt = (
            f"Assign each document below to exactly one category from the codebook, "
            f"organising by {spec['criterion']}.\n\n"
            f"CODEBOOK:\n{cbtxt}\n\n"
            f"Use category id 0 if a document fits none of them. Do not invent categories.\n"
            f"Return ONLY a JSON array with one entry per document, in order:\n"
            f'[{{"doc":1,"category":<id>}}, ...]\n\n'
            f"DOCUMENTS ({len(chunk)}):\n{body}")
        budget = 60 + 22 * len(chunk)
        got = parse_labels(ask_gpt4o(prompt, max_tokens=budget) or '')
        if got is None:
            got = parse_labels(ask_gpt4o(prompt, max_tokens=budget) or '')
        if got is not None:
            out = {chunk[p - 1]: c for p, c in got.items() if 1 <= p <= len(chunk)}
            return {i: (c if 0 <= c <= len(codebook) else -1)
                    for i, c in out.items()}, 0
        out = {}
        for i in chunk:
            try:
                c = judge.classify_document(texts[i], codebook, spec)
            except Exception:
                c = None
            out[i] = int(c) if c is not None else -1
        return out, 1

    fallbacks = 0
    rows = [{'doc_idx': i, 'category': c} for i, c in done.items()]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one_chunk, ch): ch for ch in chunks}
        for n, fut in enumerate(as_completed(futs), 1):
            got, fb = fut.result()
            fallbacks += fb
            for i in futs[fut]:
                rows.append({'doc_idx': i, 'category': got.get(i, -1)})
            if n % 5 == 0:
                pd.DataFrame(rows).to_csv(f, index=False)
    pd.DataFrame(rows).to_csv(f, index=False)
    return {r['doc_idx']: r['category'] for r in rows}, fallbacks


def main():
    score_rows = []
    if OUT_SCORES.exists():
        score_rows = pd.read_csv(OUT_SCORES).to_dict('records')
    done_pairs = {(r['run'], r['bench']) for r in score_rows}
    meta_rows = pd.read_csv(OUT_META).to_dict('records') if OUT_META.exists() else []

    benches = [b for b in sorted(CRITERIA) if not ONLY or b in ONLY]
    for run in RUNS:
        for bench in benches:
            if (run, bench) in done_pairs:
                continue
            try:
                texts = pd.read_csv(RQ1C / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
                    .fillna('').astype(str).tolist()
                codebook, clip, source = get_codebook(run, bench, texts)
                ref, fallbacks = get_reference(run, bench, texts, codebook)
            except RuntimeError:
                raise                              # budget stop: end the whole run
            except Exception as e:
                print(f'run{run} {bench}: FAILED ({type(e).__name__}: {e}) - '
                      f'continuing, will retry on resume', flush=True)
                continue
            usable = {i: c for i, c in ref.items() if c > 0}
            ref_idx = np.array(sorted(usable))
            ref_cat = np.array([usable[i] for i in ref_idx])

            part = pd.read_csv(RQ1C / 'data' / 'rq1' / 'partitions_instructor_gen' / f'{bench}.csv')
            pcols = [c for c in part.columns if c not in ('doc_idx', 'true_label')]
            for pid in pcols:
                s, n_used = score_partition(ref_idx, ref_cat, part[pid].to_numpy())
                score_rows.append({'run': run, 'bench': bench, 'partition_id': pid,
                                   'judge': s, 'n_docs_used': n_used})
            meta_rows.append({'run': run, 'bench': bench,
                              'codebook_categories': len(codebook),
                              'n_reference': len(usable), 'clip_chars': clip,
                              'codebook_source': source,
                              'fallback_batches': fallbacks})
            pd.DataFrame(score_rows).to_csv(OUT_SCORES, index=False)
            pd.DataFrame(meta_rows).drop_duplicates(['run', 'bench'], keep='last') \
              .to_csv(OUT_META, index=False)
            print(f'run{run} {bench}: {len(codebook)} cats (clip {clip}, {source}), '
                  f'{len(usable)} ref docs, {fallbacks} fallback batches, '
                  f'{len(pcols)} candidates scored | spend {gateway_spend()}', flush=True)
    print('gpt4o replicates complete')


if __name__ == '__main__':
    main()
