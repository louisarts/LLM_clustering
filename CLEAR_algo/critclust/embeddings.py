"""Embedding spaces.

Four objects are used downstream:

    Xr   raw document embeddings
    Xw   embeddings of the criterion rewrites
    Xc   the L2-normalised MEAN of Xr and Xw (an average, not a concatenation, despite the
         historical `concat` naming of the candidates built on it)
    P    one prototype per category, the mean embedding of that category's LLM-written
         exemplar documents

instructor-large, e5-large-v2 and paraphrase-mpnet run locally on CPU; everything else goes
through the gateway.
"""
import time

import numpy as np

from . import config
from .data import budget_guard
from .llm import CLIENT, _clip

_LOCAL_MODELS = {
    'instructor-large': 'hkunlp/instructor-large',
    'e5-large-v2': 'intfloat/e5-large-v2',
    'paraphrase-mpnet': 'sentence-transformers/paraphrase-mpnet-base-v2',
}
_PREFIX = {
    'instructor-large': 'Represent the sentence for topic clustering: ',
    'e5-large-v2': 'query: ',
    'paraphrase-mpnet': '',
}
_ST_CACHE = {}


def _local(model_id):
    if model_id not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_CACHE[model_id] = SentenceTransformer(_LOCAL_MODELS[model_id], device='cpu')
    return _ST_CACHE[model_id]


def embed(embedder, texts, cache):
    """L2-normalised embeddings, memoised to `cache` (a .npy path)."""
    if cache.exists():
        return np.load(cache)

    if embedder in _LOCAL_MODELS:
        prefix = _PREFIX[embedder]
        X = _local(embedder).encode([prefix + _clip(t, 2000) for t in texts],
                                    batch_size=64, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
    else:
        budget_guard()
        out = []
        # gemini-embedding-001 defaults to 1536-d through the gateway; the study's caches
        # are 3072-d, so the full dimensionality must be requested explicitly.
        kw = {'dimensions': 3072} if 'gemini' in embedder else {}
        for start in range(0, len(texts), 64):
            batch = [_clip(t, 3000) or ' ' for t in texts[start:start + 64]]
            for attempt in range(5):
                try:
                    r = CLIENT.embeddings.create(model=embedder, input=batch, **kw)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt * 3)
            out.extend([e.embedding for e in r.data])
        X = np.asarray(out, dtype=np.float32)

    X = np.asarray(X, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    np.save(cache, X)
    return X


def spaces(bench, setting, docs, rewrite_texts, exemplar_map):
    """-> (Xr, Xw, Xc, P) for this benchmark under this setting."""
    if setting == 'unmatched':
        Xr = np.load(config.GEMINI_EMB / f'{bench}_raw.npy')
        Xw = np.load(config.GLM5_EMB / f'{bench}_rew.npy')
        P = np.stack([np.load(config.GLM5_EMB / f'{bench}_ex{c}.npy').mean(0)
                      for c, texts in exemplar_map.items() if texts])
    elif setting in config.HYBRID_LLM_SRC:
        # Hybrids: the embedder comes from one source setting, the rewrite/exemplar TEXTS
        # from the other, so Xw and P need their own caches keyed by both.
        embedder, _ = config.models_for(bench, setting)
        llm_src = config.llm_source_setting(setting)
        if embedder == 'gemini-embedding-001' and llm_src == 'unmatched':
            return spaces(bench, 'unmatched', docs, rewrite_texts, exemplar_map)
        # raw documents are LLM-independent: reuse the per-embedder caches. A legacy
        # cache is only trusted when its dimensionality matches this encoder (the old
        # study's massive_intent cache, e.g., is 3584-d gemma-embed, not the proxy).
        _EXP_DIM = {'instructor-large': 768, 'e5-large-v2': 1024, 'paraphrase-mpnet': 768,
                    'text-embedding-3-small': 1536, 'gemini-embedding-001': 3072}
        if embedder == 'gemini-embedding-001':
            Xr = np.load(config.GEMINI_EMB / f'{bench}_raw.npy')
        else:
            Xr = None
            legacy = config.MATCHED_EMB / f'A_{bench}_raw.npy'
            if legacy.exists() and embedder not in ('e5-large-v2',):
                Xr = np.load(legacy)
                if Xr.shape[1] != _EXP_DIM.get(embedder, Xr.shape[1]):
                    Xr = None
            if Xr is None:
                tag0 = f'{embedder.replace("/", "_")}_{bench}'
                Xr = embed(embedder, docs, config.EMB_CACHE / f'{tag0}_raw.npy')
        tag = f'{embedder.replace("/", "_")}_{llm_src}_{bench}'
        Xw = embed(embedder, rewrite_texts, config.EMB_CACHE / f'{tag}_rew.npy')
        P = np.stack([embed(embedder, texts, config.EMB_CACHE / f'{tag}_ex{c}.npy').mean(0)
                      for c, texts in exemplar_map.items() if texts])
        Xc = (Xr + Xw) / 2
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        assert Xr.shape[1] == Xw.shape[1] == P.shape[1], (
            f'{bench}/{setting}: embedding dimension mismatch '
            f'({Xr.shape[1]}, {Xw.shape[1]}, {P.shape[1]}) - stale cache?')
        return Xr, Xw, Xc, P
    else:
        embedder = config.MATCHED_CFG[bench][0]
        legacy = config.MATCHED_EMB / f'A_{bench}_raw.npy'
        if legacy.exists() and embedder not in ('e5-large-v2',):
            # reuse the original study's cache where it exists and matches this encoder
            Xr = np.load(legacy)
            Xw = np.load(config.MATCHED_EMB / f'A_{bench}_rew.npy')
            P = np.stack([np.load(config.MATCHED_EMB / f'A_{bench}_ex{c}.npy').mean(0)
                          for c, texts in exemplar_map.items() if texts])
        else:
            tag = f'{embedder.replace("/", "_")}_{bench}'
            Xr = embed(embedder, docs, config.EMB_CACHE / f'{tag}_raw.npy')
            Xw = embed(embedder, rewrite_texts, config.EMB_CACHE / f'{tag}_rew.npy')
            P = np.stack([embed(embedder, texts, config.EMB_CACHE / f'{tag}_ex{c}.npy').mean(0)
                          for c, texts in exemplar_map.items() if texts])

    Xc = (Xr + Xw) / 2
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

    assert Xr.shape[1] == Xw.shape[1] == P.shape[1], (
        f'{bench}: embedding dimension mismatch '
        f'({Xr.shape[1]}, {Xw.shape[1]}, {P.shape[1]}) - stale cache?')
    return Xr, Xw, Xc, P
