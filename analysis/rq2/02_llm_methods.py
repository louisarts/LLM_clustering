#!/usr/bin/env python
"""RQ2-clean stage 2: the six paid k-inference methods on the 19 benchmarks (~$50).

Six methods x 10 seeds x 19 corpora, Gemini 2.5 Flash temp 0 via the Chattermill
gateway. Criteria verbatim from research_question_3_clean/criteria.json. No method
ever sees labels, gold k, or a granularity hint.

  direct      thinking: 300-doc sample, "how many categories does the FULL corpus
              need?" (given true N)
  probe       thinking: propose coarse/natural/fine codebooks; k = |natural|
  pairwise    ClusterLLM-style: Ward tree on a 2,000-doc subsample, same/different
              votes at the shared-grid merge levels, logistic transition point
  sequential  TopicGPT-style: scan 600 docs in batches of 20 growing a category
              list, stop after 3 empty batches, thinking merge; k = final size
  simpson     moment estimator: 120 random pairs, k = 1 / P(same), clipped [2,400]
  merge_conv  hybrid: k0 = 200 k-means micro-clusters (the grid maximum, same for
              every corpus), LLM ratifies closest-pair merges until a round
              accepts none; k = survivors. Round cap 60 so convergence from 200
              is attainable (no artificial floor).

Geometry (pairwise Ward, merge_conv k-means) runs on the INSTRUCTOR generic-
instruction embeddings ({bench}_instructor_gen.npy). The shared candidate-k grid
is 25 log-spaced points on [2, 200], rounded (23 unique), gold-blind and
identical for every corpus and method.

Appends to data/llm_k_runs.csv, idempotent per (bench, method, seed); pairwise
votes logged to data/pairwise_votes.csv. Budget-guarded. Usage: --run to launch.
"""
import json
import os
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

RQ2C = Path(__file__).resolve().parents[2]
ROOT = RQ2C.parent
RQ1C = ROOT / 'research_question_1_clean'
sys.path.append(str(RQ1C / 'scripts'))
sys.path.append(str(Path(__file__).resolve().parent))  # LLM_call.py sits next to this file
from judge import _ask_thinking, _clip, _parse_categories            # noqa: E402
from LLM_call import ask_llm                                         # noqa: E402

OUT = RQ2C / 'data' / 'rq2' / 'llm_k_runs.csv'
VOTES = RQ2C / 'data' / 'rq2' / 'pairwise_votes.csv'
CRITERIA = json.loads((ROOT / 'research_question_3_clean' / 'criteria.json').read_text())
BENCHES = sorted(CRITERIA)
SEEDS = list(range(10))
BUDGET_STOP = 880.0
RUN = '--run' in sys.argv

# 25 log-spaced points on [2, 200], rounded to integers (23 unique)
K_GRID = [2, 3, 4, 5, 6, 8, 9, 11, 14, 17, 20, 24, 29, 36, 43, 52, 63, 77,
          93, 112, 136, 165, 200]


def gateway_spend():
    try:
        base = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3].rstrip('/')
        req = urllib.request.Request(
            f'{base}/key/info',
            headers={'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return float(json.load(r)['info']['spend'])
    except Exception:
        return None


def load_bench(bench):
    texts = pd.read_csv(RQ1C / 'data' / 'rq2' / 'texts' / f'{bench}.csv')['text'] \
        .fillna('').astype(str).tolist()
    X = np.load(RQ1C / 'data' / 'rq2' / 'embeddings' / f'{bench}_instructor_gen.npy') \
        .astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return texts, X, CRITERIA[bench]


def sample300(texts, seed):
    rng = np.random.RandomState(1000 + seed)
    pool = rng.choice(len(texts), min(1000, len(texts)), replace=False)
    return [texts[i] for i in pool[:300]]


def retry_ask(prompt, thinking=False, tries=4):
    import time
    for a in range(tries):
        try:
            if thinking:
                reply, fin = _ask_thinking(prompt)
                if fin == 'length':
                    raise ValueError('truncated')
                return reply
            return ask_llm(prompt) or ''
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(2 ** a * 3)


# ---------------- methods ----------------------------------------------------
def k_direct(docs, crit, n_total):
    body = '\n'.join(f'{i}. {_clip(d, 400)}' for i, d in enumerate(docs, 1))
    prompt = (
        f"A corpus of {n_total} documents is to be organised into categories "
        f"by {crit}.\n\nBelow are {len(docs)} documents sampled from it. Read "
        f"them, then estimate how many categories the FULL corpus needs. "
        f"Choose the granularity you judge most natural.\n"
        f'Respond only with JSON: {{"k": <integer>}}\n\nDocuments:\n{body}'
    )
    reply = retry_ask(prompt, thinking=True)
    m = re.search(r'"k"\s*:\s*(\d+)', reply)
    if not m:
        raise ValueError('no k')
    return int(m.group(1))


def k_probe(docs, crit):
    body = '\n'.join(f'{i}. {_clip(d, 400)}' for i, d in enumerate(docs, 1))
    prompt = (
        f"A corpus of documents is to be organised into categories by {crit}. "
        f"Different granularities of codebook can be valid.\n\nBelow are "
        f"{len(docs)} documents sampled from the corpus. Propose THREE "
        f'codebooks as lists of category names (names only): "coarse", '
        f'"natural", "fine". Cover the whole corpus.\n\n'
        f'Respond only with JSON:\n{{"coarse": [...], "natural": [...], '
        f'"fine": [...]}}\n\nDocuments:\n{body}'
    )
    reply = retry_ask(prompt, thinking=True)
    m = re.search(r'\{.*\}', reply, re.DOTALL)
    obj = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
    return len(obj['natural'])


def same_category(a, b, crit):
    prompt = (
        f"Two documents from the same corpus. By {crit}, do they belong to the "
        f'SAME category?\nRespond only with JSON: {{"same": true}} or '
        f'{{"same": false}}\n\nDocument A: {_clip(a)}\n\nDocument B: {_clip(b)}'
    )
    r = retry_ask(prompt)
    m = re.search(r'"same"\s*:\s*(true|false)', r)
    return None if not m else m.group(1) == 'true'


def k_pairwise(texts, X, crit, bench, seed):
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(X), min(2000, len(X)), replace=False)
    Z = linkage(X[sub], method='ward')
    m = len(sub)
    vote_rows = []
    for k in K_GRID:
        if k >= m // 4:
            break
        votes = []
        jobs = []
        for j in range(3):
            kk = k + j
            lb = fcluster(Z, t=kk + 1, criterion='maxclust')
            la = fcluster(Z, t=kk, criterion='maxclust')
            merged = None
            for cid in np.unique(la):
                inside = np.unique(lb[la == cid])
                if len(inside) == 2:
                    merged = inside
                    break
            if merged is None:
                continue
            ga = np.where(lb == merged[0])[0]
            gb = np.where(lb == merged[1])[0]
            for _ in range(3):
                jobs.append((texts[sub[rng.choice(ga)]],
                             texts[sub[rng.choice(gb)]]))
        with ThreadPoolExecutor(max_workers=9) as pool:
            futs = [pool.submit(same_category, a, b, crit) for a, b in jobs]
            for fut in as_completed(futs):
                v = fut.result()
                if v is not None:
                    votes.append(v)
        if votes:
            vote_rows.append({'bench': bench, 'seed': seed, 'k': k,
                              'n_votes': len(votes),
                              'frac_same': float(np.mean(votes))})
    with _lock:
        pd.DataFrame(vote_rows).to_csv(VOTES, mode='a' if VOTES.exists() else 'w',
                                       header=not VOTES.exists(), index=False)
    g = pd.DataFrame(vote_rows).sort_values('k')
    xs, ys = [], []
    for _, r in g.iterrows():
        n_same = int(round(r.frac_same * r.n_votes))
        xs += [np.log(r.k)] * int(r.n_votes)
        ys += [0] * n_same + [1] * (int(r.n_votes) - n_same)
    xs, ys = np.array(xs), np.array(ys)
    if len(set(ys)) < 2:
        return int(g.k.min()) if ys.mean() < 0.5 else int(g.k.max())
    if ys.mean() < 0.5 and ys[xs <= np.log(4)].mean() < 0.5:
        return int(g.k.min())
    if ys.mean() >= 0.5 and ys[xs >= np.log(90)].mean() >= 0.5:
        return int(g.k.max())
    lr = LogisticRegression().fit(xs.reshape(-1, 1), ys)
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    if b1 >= 0:
        return int(g.k.max()) if ys.mean() >= 0.5 else int(g.k.min())
    return int(np.clip(round(np.exp(-b0 / b1)), g.k.min(), g.k.max()))


def k_sequential(texts, crit, seed):
    rng = np.random.RandomState(1000 + seed)
    pool = rng.choice(len(texts), min(600, len(texts)), replace=False)
    cats, dry = [], 0
    for s0 in range(0, len(pool), 20):
        batch = [texts[i] for i in pool[s0:s0 + 20]]
        listing = '\n'.join(f'- {c}' for c in cats) if cats else '(none yet)'
        body = '\n'.join(f'{i}. {_clip(d, 300)}' for i, d in enumerate(batch, 1))
        prompt = (
            f"A corpus is being organised into categories by {crit}. Categories "
            f"found so far:\n{listing}\n\nBelow are {len(batch)} new documents. "
            f"List ONLY the genuinely new categories (by {crit}) needed for "
            f"documents that fit none of the existing categories. If none are "
            f'needed, respond {{"new": []}}.\n'
            f'Respond only with JSON: {{"new": ["...", ...]}}\n\n'
            f'Documents:\n{body}'
        )
        r = retry_ask(prompt)
        m = re.search(r'\{.*\}', r, re.DOTALL)
        try:
            new = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))['new']
        except Exception:
            new = []
        new = [str(x).strip() for x in new if str(x).strip()]
        if new:
            cats.extend(new)
            dry = 0
        else:
            dry += 1
            if dry >= 3:
                break
    if len(cats) < 2:
        return max(len(cats), 1)
    listing = '\n'.join(f'{i}. {c}' for i, c in enumerate(cats, 1))
    prompt = (
        f"These category names were collected while organising one corpus by "
        f"{crit}. Merge duplicates and near-duplicates and return the final "
        f"deduplicated list of category names.\n"
        f'Respond only with JSON: {{"categories": ["...", ...]}}\n\n{listing}'
    )
    r = retry_ask(prompt, thinking=True)
    m = re.search(r'\{.*\}', r, re.DOTALL)
    final = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))['categories']
    return len(final)


def k_simpson(texts, crit, seed):
    rng = np.random.RandomState(1000 + seed)
    pairs = [(rng.randint(len(texts)), rng.randint(len(texts)))
             for _ in range(130)]
    pairs = [(a, b) for a, b in pairs if a != b][:120]
    votes = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(same_category, texts[a], texts[b], crit)
                for a, b in pairs]
        for fut in as_completed(futs):
            v = fut.result()
            if v is not None:
                votes.append(v)
    p_same = max(np.mean(votes), 1.0 / 400)
    return int(np.clip(round(1.0 / p_same), 2, 400))


def k_merge_conv(texts, X, crit, seed):
    k0 = 200  # the grid maximum: every grid value reachable by merging down
    km = KMeans(n_clusters=k0, n_init=3, random_state=seed).fit(X)
    lab = km.labels_

    def describe(c):
        members = np.where(lab == c)[0]
        d = ((X[members] - km.cluster_centers_[c]) ** 2).sum(1)
        central = members[np.argsort(d)[:3]]
        return ' / '.join(_clip(texts[i], 110) for i in central)

    desc = {c: describe(c) for c in range(k0)}
    parent = list(range(k0))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    centers = {c: km.cluster_centers_[c] for c in range(k0)}
    for _ in range(60):  # from k0=200, 12 rounds x 8 pairs would floor k at 104
        alive = sorted({find(c) for c in range(k0)})
        if len(alive) <= 2:
            break
        C = np.stack([centers[c] for c in alive])
        D = 1 - C @ C.T
        np.fill_diagonal(D, np.inf)
        cand = []
        seen = set()
        for i in np.argsort(D, axis=None):
            a, b = divmod(int(i), len(alive))
            key = tuple(sorted((alive[a], alive[b])))
            if key not in seen:
                seen.add(key)
                cand.append(key)
            if len(cand) >= 8:
                break

        def vote(pair):
            a, b = pair
            prompt = (
                f"A corpus is organised by {crit}. Are these two groups the "
                f"SAME category?\nGroup A examples: {desc[a]}\n"
                f"Group B examples: {desc[b]}\n"
                f'Respond only with JSON: {{"same": true}} or {{"same": false}}'
            )
            r = retry_ask(prompt)
            m = re.search(r'"same"\s*:\s*(true|false)', r)
            return pair, (None if not m else m.group(1) == 'true')

        merged_any = False
        with ThreadPoolExecutor(max_workers=8) as pool:
            for fut in as_completed([pool.submit(vote, p) for p in cand]):
                (a, b), v = fut.result()
                if v:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                        centers[ra] = (centers[ra] + centers[rb]) / 2
                        desc[ra] = desc[ra][:120] + ' / ' + desc[rb][:120]
                        merged_any = True
        if not merged_any:
            break
    return len({find(c) for c in range(k0)})


# ---------------- driver ------------------------------------------------------
METHODS = ['direct', 'probe', 'pairwise', 'sequential', 'simpson', 'merge_conv']

if not RUN:
    n = len(BENCHES) * len(SEEDS) * len(METHODS)
    print(f'dry run: {n} runs pending across {METHODS}. Relaunch with --run.')
    sys.exit(0)

runs = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(
    columns=['bench', 'method', 'seed', 'k'])
if not OUT.exists():
    runs.to_csv(OUT, index=False)
done = {(r.bench, r.method, r.seed) for r in runs.itertuples()}
DATA = {}
_lock = threading.Lock()


def get_bench(b):
    with _lock:
        if b not in DATA:
            DATA[b] = load_bench(b)
        return DATA[b]


jobs = [(b, m, s) for b in BENCHES for m in METHODS for s in SEEDS
        if (b, m, s) not in done]
print(f'{len(jobs)} runs pending | spend now: {gateway_spend()}', flush=True)


def run_job(job):
    b, m, s = job
    texts, X, crit = get_bench(b)
    if m == 'direct':
        return job, k_direct(sample300(texts, s), crit, len(texts))
    if m == 'probe':
        return job, k_probe(sample300(texts, s), crit)
    if m == 'pairwise':
        return job, k_pairwise(texts, X, crit, b, s)
    if m == 'sequential':
        return job, k_sequential(texts, crit, s)
    if m == 'simpson':
        return job, k_simpson(texts, crit, s)
    return job, k_merge_conv(texts, X, crit, s)


n_done, n_fail = 0, 0
with ThreadPoolExecutor(max_workers=6) as outer:
    futs = {outer.submit(run_job, j): j for j in jobs}
    for fut in as_completed(futs):
        j = futs[fut]
        try:
            _, k = fut.result()
        except Exception as e:
            n_fail += 1
            print(f'   !! {j}: {str(e)[:80]}', flush=True)
            continue
        with _lock:
            pd.DataFrame([{'bench': j[0], 'method': j[1], 'seed': j[2],
                           'k': int(k)}]).to_csv(OUT, mode='a', header=False,
                                                 index=False)
        n_done += 1
        if n_done % 25 == 0:
            sp = gateway_spend()
            print(f'   {n_done}/{len(jobs)} done ({n_fail} failed) | '
                  f'{"$%.2f" % sp if sp else "?"}', flush=True)
            if sp is not None and sp > BUDGET_STOP:
                print('!! BUDGET STOP'); os._exit(1)

runs = pd.read_csv(OUT)
print(runs.pivot_table(index='bench', columns='method', values='k',
                       aggfunc='count').to_string())
print(f'{n_fail} failures (rerun to sweep) | final spend: ${gateway_spend():.2f}')
