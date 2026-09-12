"""Sentence-embedding of text documents.

- embed_miniLM(documents): L2-normalized MiniLM vectors (used by experiments 1 & 2).
- embed_instructor(documents): L2-normalized INSTRUCTOR vectors, conditioned on a
  generic clustering instruction (used by experiment 3).

Importing this module also pins the numeric libraries to single-thread (before
torch / sklearn load), which keeps clustering deterministic and avoids
BLAS/OpenMP contention on macOS.
"""

import os

# default to single-threaded (determinism / BLAS contention) but respect
# values already set in the environment, e.g. for long-document embedding runs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from sklearn.preprocessing import normalize

import torch
# single-threaded by default (determinism / BLAS contention); override with
# TORCH_NUM_THREADS for long-document embedding runs where 1 thread is ~8x slower
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
from sentence_transformers import SentenceTransformer

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
INSTRUCTOR_MODEL = "hkunlp/instructor-large"
CLUSTERING_INSTRUCTION = "Represent the sentence for clustering: "


def embed_miniLM(documents, embedder=EMBEDDER):
    """Embed documents into L2-normalized MiniLM sentence vectors."""
    model = SentenceTransformer(embedder, device="cpu")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=16)
    return normalize(embeddings)


_instructor = None


def _load_instructor():
    """Load INSTRUCTOR once, configured to EXCLUDE the instruction from pooling.

    This matches the original InstructorEmbedding behavior: the instruction
    steers the representation but its tokens are not mean-pooled into the vector
    (otherwise the shared prefix inflates similarity and collapses clustering).
    """
    global _instructor
    if _instructor is None:
        model = SentenceTransformer(INSTRUCTOR_MODEL, device="cpu")
        for module in model:
            if type(module).__name__ == "Pooling":
                module.include_prompt = False
        _instructor = model
    return _instructor


def embed_instructor(documents, instruction=CLUSTERING_INSTRUCTION):
    """Embed documents with INSTRUCTOR, conditioned on `instruction` (prepended
    but excluded from pooling). Returns L2-normalized vectors."""
    model = _load_instructor()
    embeddings = model.encode([str(d) for d in documents], prompt=instruction,
                              show_progress_bar=True, batch_size=16)
    return normalize(embeddings)
