#!/usr/bin/env python
"""Embed the 19 corpora with INSTRUCTOR-large under the GENERIC instruction
"Represent the sentence(s) for clustering: " (vs the topic-flavoured instruction of the
primary bed). Local CPU only; writes data/embeddings/{bench}_instructor_gen.npy.
Resumable per benchmark."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RQ1C = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RQ1C.parent / 'research_question_3_clean'))
from critclust import config as c3  # noqa: E402

INSTRUCTION = 'Represent the sentence(s) for clustering: '

def clip(t, n=2000):
    return ' '.join(str(t).split())[:n]

from sentence_transformers import SentenceTransformer  # noqa: E402
model = None

for bench in c3.BENCH19:
    out = RQ1C / 'data' / 'rq1' / 'embeddings' / f'{bench}_instructor_gen.npy'
    if out.exists():
        print(f'{bench}: exists', flush=True)
        continue
    texts = pd.read_csv(RQ1C / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
        .fillna('').astype(str).tolist()
    if model is None:
        model = SentenceTransformer('hkunlp/instructor-large', device='cpu')
    X = model.encode([INSTRUCTION + clip(t) for t in texts], batch_size=64,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
    assert len(X) == len(texts)
    np.save(out, X)
    print(f'{bench}: n={len(X)} dim={X.shape[1]}', flush=True)
print('done')
