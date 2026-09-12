"""Label-free selection among candidate clusterings.

Both judges score with AMI against the LLM reference labelling - never gold - and AMI rather
than NMI because the reference and the candidates rarely share a granularity, and AMI is
chance-corrected so a finer partition earns no free credit.

judge_A  scores on the whole reference, then walks CAND_ORDER_A and takes the first candidate
         within TIE_EPS of the best (the less-LLM tie-break).
judge_B  scores only on a held-out 30% of the reference, averaged over B_BOOT bootstrap
         resamples that are drawn once and shared by every candidate, then takes the plain
         argmax. The split exists because three of pool_B's candidates are fitted to the
         reference and would otherwise be scored on their own training labels.
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

from . import config


def split_reference(fm, seed):
    """-> (train_ids, judge_ids). Redrawn per seed."""
    ids = np.array(sorted(fm))
    perm = np.random.RandomState(seed).permutation(len(ids))
    cut = int(config.JUDGE_SPLIT * len(ids))
    return ids[perm[:cut]], ids[perm[cut:]]


def judge_A(fm, n_docs, min_docs=50):
    """-> (score_fn, select_fn) scoring on the full reference labelling."""
    mask = np.array([i in fm for i in range(n_docs)])
    ref = np.array([fm[i] for i in np.where(mask)[0]])

    def score_fn(labels):
        if mask.sum() < min_docs:
            return np.nan
        return ami(ref, np.asarray(labels)[mask])

    def select_fn(cands):
        scores = {name: score_fn(lab) for name, lab in cands.items()}
        best = max(scores.values())
        order = config.CAND_ORDER_A + [c for c in cands if c not in config.CAND_ORDER_A]
        winner = next(name for name in order
                      if name in scores and scores[name] >= best - config.TIE_EPS)
        return winner, scores

    return score_fn, select_fn


def judge_B(fm, judge_ids, seed):
    """-> (score_fn, select_fn) scoring on the held-out ids, bootstrap-averaged.

    The B_BOOT resamples are drawn once here and reused for every candidate, making the
    comparison paired: all candidates are measured on identical resamples.
    """
    held_out = np.array([fm[i] for i in judge_ids])
    boots = [np.random.RandomState(seed * 100 + b).choice(len(judge_ids), len(judge_ids),
                                                          replace=True)
             for b in range(config.B_BOOT)]

    def score_fn(labels):
        lab = np.asarray(labels)[judge_ids]
        return float(np.mean([ami(held_out[b], lab[b]) for b in boots]))

    def select_fn(cands):
        scores = {name: score_fn(lab) for name, lab in cands.items()}
        return max(scores, key=scores.get), scores

    return score_fn, select_fn
