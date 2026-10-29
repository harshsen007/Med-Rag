"""
src/retrieve.py — Retrieve the top-k most relevant chunks for a question.

Public function:
  retrieve(question) → list[dict]

Each dict has:  text, score, source, section, page_start, page_end
(plus doc_type and chunk_index from storage)
"""

from src.config import settings
from src import vector_store


def retrieve(question: str) -> list[dict]:
    """
    Embed the question and fetch the top-k chunks from Qdrant.

    Uses SEARCH_MODE (hybrid / dense / sparse), TOP_K, and
    CANDIDATES_PER_BRANCH from config.

    Returns a list of chunk dicts sorted by relevance (highest score first).
    Each dict has at minimum: text, score, source, section, page_start, page_end.
    """
    chunks = vector_store.search(
        question=question,
        mode=settings.SEARCH_MODE,
        top_k=settings.TOP_K,
        candidates=settings.CANDIDATES_PER_BRANCH,
    )
    return chunks
