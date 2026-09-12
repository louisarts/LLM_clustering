"""Benchmark loading, clustering metrics and the gateway budget guard."""
import json
import urllib.request
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as nmi

from . import config


def load_bench(bench):
    """-> (docs, gold, k).

    k is the FULL-corpus class count. For subsampled P2P/S2S benchmarks the evaluation
    subsample may not contain every class, so len(unique(subsample_gold)) understates it.
    """
    texts_csv = config.NEWBENCH / f'{bench}_texts.csv'
    if texts_csv.exists():
        d = pd.read_csv(texts_csv)
        return d.text.astype(str).tolist(), d.gold.to_numpy(), int(d.gold.nunique())

    raise FileNotFoundError(f'no benchmark texts for {bench!r} at {texts_csv}')


def hungarian_accuracy(gold, labels):
    """Clustering accuracy under the optimal one-to-one cluster/class matching."""
    g = pd.factorize(pd.Series(gold))[0]
    l = pd.factorize(pd.Series(labels))[0]
    counts = np.zeros((l.max() + 1, g.max() + 1), int)
    np.add.at(counts, (l, g), 1)
    rows, cols = linear_sum_assignment(-counts)
    return counts[rows, cols].sum() / len(gold)


def score(gold, labels):
    """-> (NMI, ACC) as percentages, rounded. NMI is V-measure (arithmetic normalisation),
    which is what the literature we compare against reports."""
    return (round(nmi(gold, labels) * 100, 2),
            round(hungarian_accuracy(gold, labels) * 100, 2))


def onehot(labels):
    """n x n_clusters binary indicator matrix, used to build the consensus candidate."""
    f = pd.factorize(labels)[0]
    m = np.zeros((len(f), f.max() + 1), np.float32)
    m[np.arange(len(f)), f] = 1
    return m


def gateway_spend():
    """Dollars spent on the LiteLLM gateway key, or None if it cannot be read."""
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


def budget_guard():
    """Abort before any LLM call that would take spend past config.BUDGET_STOP."""
    spend = gateway_spend()
    if spend is not None and spend > config.BUDGET_STOP:
        raise RuntimeError(f'BUDGET STOP: gateway spend ${spend:.2f} > ${config.BUDGET_STOP:.2f}')
