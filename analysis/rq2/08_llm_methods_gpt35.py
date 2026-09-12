#!/usr/bin/env python
"""RQ2 gpt-3.5-turbo bed, stage 2: the six paid k-inference methods rerun with
gpt-3.5-turbo (~$15-18 projected; the third RQ2 bed).

Method code is imported verbatim from 02_llm_methods.py; only the ask layer is
rebound (no reasoning mode; the 4,096-token completion cap). The 16,385-token
context cannot always hold the 300-doc discovery prompts, so direct and probe
mirror the RQ1 gpt-3.5 bed's disclosed fallback: documents are pre-clipped to
150 characters whenever the 400-character prompt estimate overruns 12,500
input tokens (same rule and constants as 12_judge_replicates_gpt35.py).

Outputs (never touches the Flash or 4o-mini files):
    data/llm_k_runs_gpt35.csv        bench, method, seed, k
    data/pairwise_votes_gpt35.csv

Usage:  --probe   banking77 x seed 0 x all six methods, prints measured spend
        --run     full sweep, cheapest method first, budget-guarded at $999.
"""
import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import pandas as pd
from openai import OpenAI

RQ2C = Path(__file__).resolve().parents[2]
HERE = RQ2C / 'scripts'
OUT = RQ2C / 'data' / 'rq2' / 'llm_k_runs_gpt35.csv'
BUDGET_STOP = 999.0
MODEL = 'gpt-3.5-turbo'
INPUT_BUDGET_TOKENS = 12500     # discovery prompt budget inside the 16,385 context
PROBE = '--probe' in sys.argv
RUN = '--run' in sys.argv

_argv = sys.argv
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location('m02', HERE / '02_llm_methods.py')
m02 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m02)
except SystemExit:
    pass
sys.argv = _argv

m02.VOTES = RQ2C / 'data' / 'rq2' / 'pairwise_votes_gpt35.csv'
m02._lock = threading.Lock()

CLIENT = OpenAI(base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'], timeout=240, max_retries=2)


def retry_ask_gpt35(prompt, thinking=False, tries=4):
    """No reasoning mode; completions cap at 4,096 tokens. Near the key's cap
    the gateway intermittently 429s concurrent bursts - wait those out."""
    max_tokens = 4096 if thinking else 2000
    budget_429 = 0
    a = 0
    while True:
        try:
            r = CLIENT.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}])
            if thinking and r.choices[0].finish_reason == 'length':
                raise ValueError('truncated')
            return r.choices[0].message.content or ''
        except Exception as e:
            if 'Budget has been exceeded' in str(e) or '429' in str(e)[:60]:
                budget_429 += 1
                if budget_429 > 10:
                    raise
                time.sleep(30)
                continue
            a += 1
            if a >= tries:
                raise
            time.sleep(2 ** a * 3)


m02.retry_ask = retry_ask_gpt35


def discovery_docs(texts, seed):
    """The 300-doc sample, pre-clipped to 150 chars when the 400-char prompt
    would overrun the context (the RQ1 gpt-3.5 bed's disclosed fallback).
    Pre-clipping is a no-op for the later _clip(d, 400) inside the methods."""
    docs = m02.sample300(texts, seed)
    est = sum(min(len(d), 400) for d in docs) / 4
    if est > INPUT_BUDGET_TOKENS:
        docs = [d[:150] for d in docs]
    return docs


METHOD_ORDER = ['direct', 'probe', 'sequential', 'simpson', 'pairwise',
                'merge_conv']


def run_one(bench, method, seed):
    texts, X, crit = m02.load_bench(bench)
    if method == 'direct':
        return m02.k_direct(discovery_docs(texts, seed), crit, len(texts))
    if method == 'probe':
        return m02.k_probe(discovery_docs(texts, seed), crit)
    if method == 'pairwise':
        return m02.k_pairwise(texts, X, crit, bench, seed)
    if method == 'sequential':
        return m02.k_sequential(texts, crit, seed)
    if method == 'simpson':
        return m02.k_simpson(texts, crit, seed)
    return m02.k_merge_conv(texts, X, crit, seed)


if PROBE:
    bench = 'banking77'
    s0 = m02.gateway_spend()
    print(f'probe on {bench} seed 0 | spend now ${s0:.2f}', flush=True)
    total = 0.0
    for method in METHOD_ORDER:
        t0 = time.time()
        k = run_one(bench, method, 0)
        time.sleep(8)
        s1 = m02.gateway_spend()
        d = (s1 - s0) if (s0 is not None and s1 is not None) else float('nan')
        total += d
        print(f'  {method:12s} k={k:<4d} ${d:.4f}  ({time.time()-t0:.0f}s) '
              f'-> x190 ~ ${d*190:.2f}', flush=True)
        s0 = s1
    print(f'probe total ${total:.4f} | projected full sweep ~ ${total*190:.2f}',
          flush=True)
    sys.exit(0)

if not RUN:
    print('dry run. --probe for the one-bench cost probe, --run for the sweep.')
    sys.exit(0)

runs = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(
    columns=['bench', 'method', 'seed', 'k'])
if not OUT.exists():
    OUT.parent.mkdir(exist_ok=True)
    runs.to_csv(OUT, index=False)
done = {(r.bench, r.method, r.seed) for r in runs.itertuples()}
_out_lock = threading.Lock()

n_done, n_fail = 0, 0
for method in METHOD_ORDER:
    jobs = [(b, s) for b in m02.BENCHES for s in m02.SEEDS
            if (b, method, s) not in done]
    if not jobs:
        continue
    sp = m02.gateway_spend()
    print(f'== {method}: {len(jobs)} runs pending | spend '
          f'{"$%.2f" % sp if sp else "?"}', flush=True)
    if sp is not None and sp > BUDGET_STOP:
        print('!! BUDGET STOP before phase'); sys.exit(1)

    def do(job, method=method):
        b, s = job
        return job, run_one(b, method, s)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(do, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                _, k = fut.result()
            except Exception as e:
                n_fail += 1
                print(f'   !! {method} {j}: {str(e)[:80]}', flush=True)
                continue
            with _out_lock:
                pd.DataFrame([{'bench': j[0], 'method': method, 'seed': j[1],
                               'k': int(k)}]).to_csv(OUT, mode='a',
                                                     header=False, index=False)
            n_done += 1
            if n_done % 15 == 0:
                sp = m02.gateway_spend()
                print(f'   {n_done} done ({n_fail} failed) | '
                      f'{"$%.2f" % sp if sp else "?"}', flush=True)
                if sp is not None and sp > BUDGET_STOP:
                    print('!! BUDGET STOP'); os._exit(1)

runs = pd.read_csv(OUT)
print(runs.pivot_table(index='bench', columns='method', values='k',
                       aggfunc='count').to_string())
print(f'{n_fail} failures (rerun to sweep) | final spend: '
      f'${m02.gateway_spend():.2f}')
