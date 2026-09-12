"""Candidate clusterings.

pool_A  the five V0 candidates, all unsupervised.
pool_B  the fourteen V4 candidates: those five, plus consensus, plus three fitted to the
        reference labels (seeded, LDA, contrastive head), plus six unsupervised diversity
        candidates spanning different cluster shapes and dimensionalities.

The supervised candidates in pool_B see only the training portion of the reference. Nothing
in either pool touches gold labels; k is supplied by the caller.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture

from . import config
from .data import onehot

torch.set_num_threads(4)


# ------------------------------------------------------------------ contrastive head
class Head(nn.Module):
    """Shallow projection over frozen features: d -> 256 -> ReLU -> 128, L2-normalised."""

    def __init__(self, d, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def supcon(z, labels, temperature=0.1):
    """Supervised contrastive loss.

    For each anchor, the similarities to all other documents are softmaxed, and the loss is
    the negative mean log-probability mass landing on documents that share the anchor's label.
    Anchors whose category has no other member are skipped.
    """
    n = z.shape[0]
    sim = z @ z.T / temperature
    labels = labels.view(-1, 1)
    eye = torch.eye(n)
    pos = (labels == labels.T).float() * (1 - eye)
    logits = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(logits) * (1 - eye)
    log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    n_pos = pos.sum(1)
    mask = n_pos > 0
    return -((pos * log_prob).sum(1)[mask] / n_pos[mask]).mean()


def train_head(X_train, y_train, X_val, y_val, seed, epochs=300, patience=30):
    """Full-batch Adam with early stopping on the validation contrastive loss."""
    torch.manual_seed(seed)
    head = Head(X_train.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X_train, y_train, X_val, y_val = map(torch.tensor, (X_train, y_train, X_val, y_val))

    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        supcon(head(X_train), y_train).backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val = supcon(head(X_val), y_val).item()
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        head.load_state_dict(best_state)
    return head


# ------------------------------------------------------------------ pools
def _prototype_init(P, X, k, seed):
    """Starting centroids from the exemplar prototypes, topped up if there are fewer
    than k. This is the ORIGINAL rule, kept verbatim for CritClust_A and CritClust_B
    so their recorded results stay reproducible: the fill is an unconditioned k-means
    on the whole corpus."""
    if len(P) >= k:
        return KMeans(k, n_init=3, random_state=seed).fit(P).cluster_centers_
    extra = KMeans(k - len(P), n_init=3, random_state=seed).fit(X).cluster_centers_
    return np.vstack([P, extra])


def _prototype_init_pp(P, X, k, seed):
    """CritClust_C's starting centroids.

    More prototypes than k: condense them to k centres with a small k-means over the
    prototypes themselves (same as _prototype_init). Fewer than k: keep every
    prototype and fill the remaining slots k-means++-style CONDITIONED on the
    prototypes - each new centre is sampled from the documents with probability
    proportional to its squared distance from the nearest already-chosen centre, so
    the fill lands where the codebook has no coverage instead of re-covering the
    dense regions the prototypes already mark.
    """
    if len(P) >= k:
        return KMeans(k, n_init=3, random_state=seed).fit(P).cluster_centers_
    rng = np.random.RandomState(seed)
    centres = list(P)
    d2 = np.full(len(X), np.inf)
    for p in P:
        d2 = np.minimum(d2, ((X - p) ** 2).sum(1))
    for _ in range(k - len(P)):
        prob = d2 / d2.sum() if d2.sum() > 0 else None
        i = rng.choice(len(X), p=prob)
        centres.append(X[i])
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.asarray(centres)


def pool_A(Xr, Xc, P, k, seed):
    """The five V0 candidates. -> (labels_by_name, space_by_name)."""
    Zr = PCA(50, random_state=seed).fit_transform(Xr)
    Zc = PCA(50, random_state=seed).fit_transform(Xc)

    cands = {
        'kmeans_raw': KMeans(k, n_init=3, random_state=seed).fit_predict(Xr),
        'gmm_raw': GaussianMixture(k, covariance_type='diag', random_state=seed,
                                   max_iter=200).fit(Zr).predict(Zr),
        'gmm_concat': GaussianMixture(k, covariance_type='diag', random_state=seed,
                                      max_iter=200).fit(Zc).predict(Zc),
        'exemplar_raw': KMeans(k, init=_prototype_init(P, Xr, k, seed), n_init=1,
                               random_state=seed).fit_predict(Xr),
        'exemplar_concat': KMeans(k, init=_prototype_init(P, Xc, k, seed), n_init=1,
                                  random_state=seed).fit_predict(Xc),
    }
    spaces = {'kmeans_raw': Xr, 'gmm_raw': Zr, 'gmm_concat': Zc,
              'exemplar_raw': Xr, 'exemplar_concat': Xc}
    return cands, spaces


def pool_C(Xr, Xw, Xc, P, k, seed):
    """CritClust_C: pool_A's five plus three judge-independent extras, using pool_B's
    exact candidate definitions for comparability. Chosen for diversity the A pool
    lacks - km_rewrite clusters the criterion-distilled rewrite space, agglo_raw is
    the only non-centroid family, km_concat completes the averaged-space grid. None
    of them sees the reference labelling, so judging stays non-circular on the full
    reference (no train/judge split needed)."""
    cands, spaces = pool_A(Xr, Xc, P, k, seed)
    cands, spaces = dict(cands), dict(spaces)
    # C's exemplar-init candidates use the k-means++-style conditioned top-up
    cands['exemplar_raw'] = KMeans(k, init=_prototype_init_pp(P, Xr, k, seed),
                                   n_init=1, random_state=seed).fit_predict(Xr)
    cands['exemplar_concat'] = KMeans(k, init=_prototype_init_pp(P, Xc, k, seed),
                                      n_init=1, random_state=seed).fit_predict(Xc)
    Zr = PCA(50, random_state=seed).fit_transform(Xr)
    cands['km_concat'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Xc)
    cands['km_rewrite'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Xw)
    cands['agglo_raw'] = AgglomerativeClustering(k, linkage='ward').fit_predict(Zr)
    spaces.update({'km_concat': Xc, 'km_rewrite': Xw, 'agglo_raw': Zr})
    return cands, spaces


def pool_B(Xr, Xw, Xc, P, k, seed, fm, train_ids):
    """The fourteen V4 candidates. Supervised ones are fitted on `train_ids` only."""
    n_pca = min(256, len(train_ids) - 1, Xc.shape[1])
    Xp = PCA(n_pca, random_state=seed).fit_transform(Xc).astype(np.float32)
    Zr = PCA(50, random_state=seed).fit_transform(Xr)
    Zc = PCA(50, random_state=seed).fit_transform(Xc)

    cands, spaces = pool_A(Xr, Xc, P, k, seed)
    spaces = dict(spaces)

    # consensus: cluster the five base partitions' co-assignment pattern
    cands['consensus'] = KMeans(k, n_init=3, random_state=seed).fit_predict(
        np.hstack([onehot(cands[m]) for m in config.CAND_ORDER_A]))
    spaces['consensus'] = Xc

    # reference labels, factorised over the training split
    lut = pd.factorize(np.array([fm[i] for i in train_ids]))[1]
    to_idx = {v: i for i, v in enumerate(lut)}
    train_lab = np.array([to_idx[fm[i]] for i in train_ids])

    # seeded: start k-means from each reference category's mean position
    cats = np.unique(train_lab)
    centres = np.stack([Xc[train_ids[train_lab == c]].mean(0) for c in cats])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-12
    if len(centres) < k:
        extra = KMeans(k - len(centres), n_init=2, random_state=seed).fit(Xc).cluster_centers_
        centres = np.vstack([centres, extra])
    cands['seeded'] = KMeans(k, init=centres[:k], n_init=1, random_state=seed).fit_predict(Xc)
    spaces['seeded'] = Xc

    # lda: reproject to separate the reference categories, then cluster
    try:
        n_comp = min(len(np.unique(train_lab)) - 1, Xp.shape[1])
        projected = LDA(solver='eigen', shrinkage='auto', n_components=n_comp) \
            .fit(Xp[train_ids], train_lab).transform(Xp)
        cands['lda'] = KMeans(k, n_init=3, random_state=seed).fit_predict(projected)
        spaces['lda'] = projected
    except Exception:
        cands['lda'] = cands['kmeans_raw']
        spaces['lda'] = Xr

    # unsupervised diversity: different spaces and different cluster shapes
    cands['km_concat'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Xc)
    cands['km_rewrite'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Xw)
    cands['agglo_concat'] = AgglomerativeClustering(k, linkage='ward').fit_predict(Zc)
    cands['agglo_raw'] = AgglomerativeClustering(k, linkage='ward').fit_predict(Zr)
    cands['bisect_raw'] = BisectingKMeans(k, random_state=seed).fit_predict(Zr)
    cands['km_pca50'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Zc)
    spaces.update({'km_concat': Xc, 'km_rewrite': Xw, 'agglo_concat': Zc,
                   'agglo_raw': Zr, 'bisect_raw': Zr, 'km_pca50': Zc})

    # head: train on a sub-split inside train, then cluster its output for every document
    m = len(train_ids)
    fit_ids, val_ids = train_ids[:int(0.71 * m)], train_ids[int(0.71 * m):]
    lut2 = pd.factorize(np.array([fm[i] for i in fit_ids]))[1]
    to_idx2 = {v: i for i, v in enumerate(lut2)}
    fit_lab = np.array([to_idx2[fm[i]] for i in fit_ids])
    val_lab = np.array([to_idx2.get(fm[i], -1) for i in val_ids])
    keep = val_lab >= 0
    try:
        head = train_head(Xp[fit_ids], fit_lab, Xp[val_ids[keep]], val_lab[keep], seed)
        with torch.no_grad():
            Xh = head(torch.tensor(Xp, dtype=torch.float32)).numpy()
        cands['head'] = KMeans(k, n_init=3, random_state=seed).fit_predict(Xh)
        spaces['head'] = Xh
    except Exception:
        pass

    return cands, spaces
