"""Iterated judge-stopped boundary repair, used by BOTH algorithms.

V0 had this stage; V4 dropped it, but only because the ablation harness it was designed in
was cache-only and could not afford live LLM calls. Nothing showed it to be harmful - the
technique league rated iterated repair the strongest single technique measured - so
CritClust_B re-introduces it.

Each cluster is named once by the LLM and the names are frozen. Then, for up to REPAIR_ROUNDS
rounds, the documents with the smallest margin between their two nearest centroids are
re-classified against those names, and a round is kept only if the judge score improves. For
CritClust_B the judge scores on the held-out split, so repair is never validated on documents
any supervised candidate was trained on.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.spatial.distance import cdist

from . import config
from .llm import classify_batch, _clip


def name_clusters(docs, labels, X, k, criterion, ask, cache_file):
    """One short LLM-written name per cluster, from its 8 most central documents. Cached."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    means = np.stack([X[labels == c].mean(0) if (labels == c).any() else np.zeros(X.shape[1])
                      for c in range(k)])

    def name_one(c):
        members = np.where(labels == c)[0]
        if not len(members):
            return c, 'empty'
        dist = ((X[members] - means[c]) ** 2).sum(1)
        body = '\n'.join(f'- {_clip(docs[i], 250)}' for i in members[np.argsort(dist)[:8]])
        prompt = (f"Documents grouped by {criterion}. Give ONE short category name "
                  f"(2-6 words) by {criterion}. Name only.\n\n{body}")
        try:
            return c, (ask(prompt) or 'unnamed').strip()[:80]
        except Exception:
            return c, 'unnamed'

    names = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for fut in as_completed([pool.submit(name_one, c) for c in range(k)]):
            c, name = fut.result()
            names[str(c)] = name
    cache_file.write_text(json.dumps(names, indent=1))
    return names


def boundary_repair(docs, labels, X, k, criterion, spec, score_fn, ask, tag):
    """-> (labels, rounds_kept, n_llm_calls).

    The classify cache is per (algorithm, setting, benchmark) and shared across seeds, so
    later seeds re-classify only documents earlier seeds did not reach.
    """
    # The names and classify caches are only valid for the exact winner labelling they
    # were built from. A fingerprint of the incoming labels guards reuse: matching
    # fingerprint -> reuse the caches (and their paid LLM calls); mismatch (the winner
    # changed, e.g. after a pool fix) -> switch to fingerprint-suffixed cache files so
    # stale names are never applied to a different clustering.
    import zlib
    fp = zlib.crc32(np.asarray(labels, dtype=np.int64).tobytes()) & 0xffffffff
    fp_file = config.REPAIR_CACHE / f'{tag}_labels.crc'
    legacy_names = config.REPAIR_CACHE / f'{tag}_names.json'
    verified = fp_file.exists() and fp_file.read_text().strip() == str(fp)
    if not verified and legacy_names.exists():
        # cache exists for a different (or unverifiable pre-fingerprint) labelling
        tag = f'{tag}_fp{fp}'
        fp_file = config.REPAIR_CACHE / f'{tag}_labels.crc'
    fp_file.write_text(str(fp))
    cache_file = config.REPAIR_CACHE / f'{tag}_classify.json'
    names_file = config.REPAIR_CACHE / f'{tag}_names.json'
    cache = ({int(a): int(b) for a, b in json.loads(cache_file.read_text()).items()}
             if cache_file.exists() else {})

    names = name_clusters(docs, labels, X, k, criterion, ask, names_file)
    codebook = [{'id': int(c) + 1, 'name': name, 'definition': ''}
                for c, name in sorted(names.items(), key=lambda kv: int(kv[0]))]

    lab = np.asarray(labels).copy()
    means = np.stack([X[lab == c].mean(0) if (lab == c).any() else np.zeros(X.shape[1])
                      for c in range(k)])

    best_lab, best_score = lab.copy(), score_fn(lab)
    rounds_kept, n_calls, touched = 0, 0, set()

    for rnd in range(config.REPAIR_ROUNDS):
        d2 = cdist(X, means, metric='sqeuclidean')
        nearest = np.sort(d2, axis=1)
        margin_order = np.argsort(nearest[:, 1] - nearest[:, 0])
        flagged = [int(i) for i in margin_order if i not in touched][:config.REPAIR_DOSE]

        fresh = [i for i in flagged if i not in cache]
        if fresh:
            got, calls = classify_batch(fresh, docs, codebook, spec, ask,
                                        label=f'{tag} repair r{rnd + 1}')
            n_calls += calls
            for i in fresh:
                cache[i] = int(got.get(i, -1))

        for i in flagged:
            touched.add(i)
            cat = cache.get(i, -1)
            if 1 <= cat <= k:
                lab[i] = cat - 1

        means = np.stack([X[lab == c].mean(0) if (lab == c).any() else means[c]
                          for c in range(k)])
        s = score_fn(lab)
        if s > best_score:
            best_score, best_lab = s, lab.copy()
            rounds_kept += 1

    cache_file.write_text(json.dumps({str(a): b for a, b in cache.items()}))
    return best_lab, rounds_kept, n_calls
