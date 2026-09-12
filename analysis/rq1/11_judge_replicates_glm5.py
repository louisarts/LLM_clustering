#!/usr/bin/env python
"""Ten independent judge runs with the judge LLM set to glm-5, in the exact batched
configuration CritClust's unmatched judge uses - certifying the two deviations
disclosed in research_question_3_clean/results/ALIGNED_PROTOCOL.md (judge LLM glm-5
instead of the validated Flash instance; batched assignment instead of RQ1's one
call per document).

Protocol mirrors 09_judge_replicates.py run for run (same discovery/assignment
sample seeds, N_DISCOVERY=300, N_ASSIGN=1000, scoring identical):

  codebook    runs 2-10 fresh per (run, benchmark) through judge.build_codebook with
              thinking DISABLED (disclosed deviation from RQ1's default-reasoning
              codebook call: glm-5 with thinking on regularly overruns the 60k token
              limit, truncating the reply at ~$0.19 per failed attempt; thinking-off
              matches how critclust.llm.make_ask configures glm-5 everywhere else).
              Run 1 REUSES the cached glm-5 single-shot
              codebooks CritClust's unmatched setting judges with
              (research_question_3_clean/artifacts/codebooks/A_{bench}_unmatched_codebook.json),
              so the literal instrument instance is inside the validation bed.
  assignment  batched 20 docs/call, thinking off, prompt and parsing verbatim from
              critclust.llm.classify_batch (retry once, then per-document fallback).
  scoring     AMI vs the instructor_gen partition bed (free, judge.score_partition).

Outputs:
    data/replicates_glm5/run{r}/{bench}_codebook.json / {bench}_reference.csv
    results/judge_replicates_instructor_gen_glm5.csv   run, bench, partition_id, judge, n_docs_used
    results/replicate_meta_glm5.csv                    run, bench, codebook_categories,
                                                       n_reference, codebook_source, fallback_batches

Budget-guarded (hard stop $899), checkpointed per batch, resumable. Run-major loop
order so an early stop still leaves complete runs behind.
"""
import json
import os
import re
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

GLM5 = 'vertex_ai/zai-org/glm-5-maas'
RUNS = [int(x) for x in os.environ.get('RUNS', ','.join(map(str, range(1, 11)))).split(',')]
ONLY = [b for b in os.environ.get('BENCH_ONLY', '').split(',') if b]
N_DISCOVERY, N_ASSIGN = 300, 1000
BATCH, DOC_CLIP, WORKERS = 20, 1000, 8
BUDGET_STOP = 899.0

REP = RQ1C / 'data' / 'rq1' / 'replicates_glm5'
RQ3_CB = RQ1C.parent / 'research_question_3_clean' / 'artifacts' / 'codebooks'
OUT_SCORES = RQ1C / 'results' / 'rq1' / 'judge_replicates_instructor_gen_glm5.csv'
OUT_META = RQ1C / 'results' / 'rq1' / 'replicate_meta_glm5.csv'

CLIENT = OpenAI(base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'], timeout=240, max_retries=2)


def ask_glm5(prompt, max_tokens=2000, retries=4):
    """Thinking-off glm-5 call, parameters as critclust.llm.make_ask."""
    for attempt in range(retries):
        try:
            r = CLIENT.chat.completions.create(
                model=GLM5, temperature=0, max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}],
                extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            return r.choices[0].message.content or ''
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


def _ask_codebook_glm5(prompt, max_tokens=16000, retries=3):
    """Replaces judge._ask_thinking for the fresh codebooks: glm-5, thinking off.
    Returns (text, finish_reason) as build_codebook expects."""
    for attempt in range(retries):
        try:
            r = CLIENT.chat.completions.create(
                model=GLM5, temperature=0, max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}],
                extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            return (r.choices[0].message.content or ''), (r.choices[0].finish_reason or '')
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


# point judge.py's helpers at glm-5: build_codebook goes through _ask_thinking
# (swapped for the thinking-off variant above); classify_document uses judge.ask_llm.
judge.MODEL = GLM5
judge.ask_llm = ask_glm5
judge._ask_thinking = _ask_codebook_glm5


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
    """Verbatim from critclust.llm: [{"doc":1,"category":3},...] or bare [3,0,...]."""
    try:
        arr = json.loads(txt[txt.find('['): txt.rfind(']') + 1])
    except Exception:
        return None
    out = {}
    for pos, item in enumerate(arr, start=1):
        if isinstance(item, dict):
            doc, cat = item.get('doc'), item.get('category')
            if doc is None or cat is None:
                continue
            out[int(doc)] = int(cat)
        elif isinstance(item, (int, float)):
            out[pos] = int(item)
    return out or None


def get_codebook(run, bench, texts):
    f = REP / f'run{run}' / f'{bench}_codebook.json'
    if f.exists():
        return json.loads(f.read_text())['categories']
    f.parent.mkdir(parents=True, exist_ok=True)
    spec = CRITERIA[bench]
    if run == 1:                                   # the CritClust instrument instance
        cats = json.loads((RQ3_CB / f'A_{bench}_unmatched_codebook.json').read_text())
        if isinstance(cats, dict):
            cats = cats['categories']
        source = 'rq3_cached'
    else:
        guard()
        for attempt in range(3):                   # truncation retries, fresh sub-sample
            rng = np.random.RandomState(1000 * run + attempt)
            idx = rng.choice(len(texts), min(N_DISCOVERY, len(texts)), replace=False)
            try:
                cats = judge.build_codebook([texts[i] for i in idx], spec)
                break
            except ValueError as e:
                print(f'  run{run} {bench}: codebook attempt {attempt + 1} failed ({e})',
                      flush=True)
        else:
            raise RuntimeError(f'run{run} {bench}: codebook failed 3 times')
        source = 'fresh'
    f.write_text(json.dumps({'bench': bench, 'run': run, 'llm': GLM5, 'source': source,
                             'criterion': spec['criterion'], 'categories': cats}, indent=2))
    return cats


def get_reference(run, bench, texts, codebook):
    """Batched glm-5 labelling of the run's fresh 1,000-doc sample. -> (labels, fallbacks)."""
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
        got = parse_labels(ask_glm5(prompt, max_tokens=budget) or '')
        if got is None:
            got = parse_labels(ask_glm5(prompt, max_tokens=budget) or '')
        if got is not None:
            out = {chunk[p - 1]: c for p, c in got.items() if 1 <= p <= len(chunk)}
            return {i: (c if 0 <= c <= len(codebook) else -1)
                    for i, c in out.items()}, 0
        out = {}
        for i in chunk:                            # per-document fallback, as critclust
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
            texts = pd.read_csv(RQ1C / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
                .fillna('').astype(str).tolist()
            codebook = get_codebook(run, bench, texts)
            ref, fallbacks = get_reference(run, bench, texts, codebook)
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
                              'n_reference': len(usable),
                              'codebook_source': 'rq3_cached' if run == 1 else 'fresh',
                              'fallback_batches': fallbacks})
            pd.DataFrame(score_rows).to_csv(OUT_SCORES, index=False)
            pd.DataFrame(meta_rows).drop_duplicates(['run', 'bench'], keep='last') \
              .to_csv(OUT_META, index=False)
            print(f'run{run} {bench}: {len(codebook)} cats, {len(usable)} ref docs, '
                  f'{fallbacks} fallback batches, {len(pcols)} candidates scored | '
                  f'spend {gateway_spend()}', flush=True)
    print('glm5 replicates complete')


if __name__ == '__main__':
    main()
