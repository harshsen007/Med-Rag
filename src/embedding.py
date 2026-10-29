"""
src/embedding.py — Dense and sparse embeddings via fastembed.

Two module-level singletons are created on first use (lazy loading):
  _dense_model   — TextEmbedding        (BAAI/bge-small-en-v1.5 by default)
  _sparse_model  — SparseTextEmbedding  (Qdrant/bm25 by default)

Public functions:
  embed_dense(texts)  → list of numpy arrays (one per text)
  embed_sparse(texts) → list of SparseEmbedding objects (one per text)
"""

from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import settings

# Lazy singletons — loaded once, reused everywhere
_dense_model  = None
_sparse_model = None


def _get_dense() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        print(f"Loading dense model: {settings.DENSE_MODEL}")
        _dense_model = TextEmbedding(model_name=settings.DENSE_MODEL)
    return _dense_model


def _get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        print(f"Loading sparse model: {settings.SPARSE_MODEL}")
        _sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)
    return _sparse_model


def embed_dense(texts: list[str]) -> list:
    """
    Embed a list of texts with the dense model.
    Returns a list of numpy arrays (one vector per text).
    """
    return list(_get_dense().embed(texts))


def embed_sparse(texts: list[str]) -> list:
    """
    Embed a list of texts with the sparse (BM25) model.
    Returns a list of SparseEmbedding objects (one per text).
    """
    return list(_get_sparse().embed(texts))
