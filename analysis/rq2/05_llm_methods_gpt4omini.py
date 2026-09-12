#!/usr/bin/env python
"""RQ2 gpt-4o-mini bed, stage 2: the six paid k-inference methods rerun with
gpt-4o-mini as the LLM (~$20-30 projected; the RQ2 robustness replication).

Method code is imported verbatim from 02_llm_methods.py — only the ask layer is
rebound to gpt-4o-mini (no reasoning mode: thinking calls become plain calls with
the 16,384-token completion cap, mirroring the RQ1 gpt-4o-mini judge bed). Same
grid, seeds, criteria, prompts, and embeddings as the Flash bed.

Outputs (never touches the Flash files):
    data/llm_k_runs_gpt4omini.csv        bench, method, seed, k
    data/pairwise_votes_gpt4omini.csv

Usage:  --probe   one bench (banking77) x seed 0 x all six methods, prints the
                  measured spend per method and the x190 projection, then stops.
        --run     full sweep, cheapest method first, budget-guarded at $999.
"""
import importlib.util
import json
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
OUT = RQ2C / 'data' / 'rq2' / 'llm_k_runs_gpt4omini.csv'
BUDGET_STOP = 999.0
MODEL = 'gpt-4o-mini'
PROBE = '--probe' in sys.argv
RUN = '--run' in sys.argv

# import 02_llm_methods.py as a module WITHOUT triggering its driver: scrub argv
# so its RUN flag is False (it sys.exit(0)s at the dry-run gate, which we catch).
_argv = sys.argv
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location('m02', HERE / '02_llm_methods.py')
m02 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m02)
except SystemExit:
    pass
sys.argv = _argv

# rebind the pieces the method functions reach through module globals
m02.VOTES = RQ2C / 'data' / 'rq2' / 'pairwise_votes_gpt4omini.csv'
m02._lock = threading.Lock()

CLIENT = OpenAI(base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'], timeout=240, max_retries=2)


def retry_ask_4omini(prompt, thinking=False, tries=4):
    """gpt-4o-mini has no reasoning mode: thinking calls become plain calls with
    the model's 16,384-token completion cap; a length-truncated quality-critical
    reply raises so the retry/sweep machinery treats it like the Flash bed did.

    Near the key's budget cap the gateway intermittently 429s concurrent bursts
    ("Budget has been exceeded") while single calls still pass — those retries
    wait long and get extra attempts instead of burning the normal backoff."""
    max_tokens = 16384 if thinking else 2000
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


m02.retry_ask = retry_ask_4omini

METHOD_ORDER = ['direct', 'probe', 'sequential', 'simpson', 'pairwise',
                'merge_conv']


def run_one(bench, method, seed):
    texts, X, crit = m02.load_bench(bench)
    if method == 'direct':
        return m02.k_direct(m02.sample300(texts, seed), crit, len(texts))
    if method == 'probe':
        return m02.k_probe(m02.sample300(texts, seed), crit)
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
        time.sleep(8)                      # let the gateway ledger catch up
        s1 = m02.gateway_spend()
        d = (s1 - s0) if (s0 is not None and s1 is not None) else float('nan')
        total += d
        print(f'  {method:12s} k={k:<4d} ${d:.4f}  ({time.time()-t0:.0f}s) '
              f'-> x190 ~ ${d*190:.2f}', flush=True)
        s0 = s1
    print(f'probe total ${total:.4f} | projected full sweep ~ ${total*190:.2f} '
          f'(190 runs/method; probe rows already count toward the sweep)',
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
for method in METHOD_ORDER:                     # cheapest first: a budget stop
    jobs = [(b, s) for b in m02.BENCHES for s in m02.SEEDS   # loses only the
            if (b, method, s) not in done]                   # dearest methods
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

    with ThreadPoolExecutor(max_workers=3) as pool:   # burst-shy near the cap
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
