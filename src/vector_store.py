"""
src/vector_store.py — Everything related to the Qdrant collection.

Collection layout:
  - Two named vectors per point:
      "dense"  — cosine distance, size = DENSE_DIM
      "sparse" — BM25 sparse vector (requires Modifier.IDF for Qdrant/bm25)
  - Payload fields = all chunk metadata (text, source, section, …)
  - Point ID = deterministic integer hash of (source + chunk_index)
    so re-ingesting a file overwrites the old points cleanly.

Public functions:
  ensure_collection()                    — create the collection if it does not exist
  delete_by_source(source)               — remove all points for one file
  upsert_chunks(chunks)                  — embed + upsert a list of chunk dicts
  search(question, mode, top_k, candidates_per_branch) → list of chunk dicts with score
"""

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    Query,
    NearestQuery,
)
from src.config import settings
from src import embedding as emb

# One shared client instance
_client = QdrantClient(url=settings.QDRANT_URL)


# ── Collection management ─────────────────────────────────────────────────────

def ensure_collection():
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in _client.get_collections().collections]
    if settings.COLLECTION_NAME in existing:
        return  # already there

    print(f"Creating collection '{settings.COLLECTION_NAME}' ...")
    _client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=settings.DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,   # required for Qdrant/bm25
            )
        },
    )


def delete_by_source(source: str):
    """Delete all points whose payload.source == source."""
    _client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )


# ── Point ID ──────────────────────────────────────────────────────────────────

def _point_id(source: str, chunk_index: int) -> int:
    """Stable integer ID based on source filename + chunk index."""
    key = f"{source}::{chunk_index}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**63)


# ── Upserting ─────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict]):
    """
    Embed all chunks and upsert them into Qdrant.
    Processes in one batch for efficiency.
    """
    texts = [c["text"] for c in chunks]

    dense_vecs  = emb.embed_dense(texts)
    sparse_vecs = emb.embed_sparse(texts)

    points = []
    for chunk, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        point_id = _point_id(chunk["source"], chunk["chunk_index"])

        points.append(PointStruct(
            id=point_id,
            vector={
                "dense":  dv.tolist(),
                "sparse": SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                ),
            },
            payload=chunk,   # store everything — text, source, section, pages, …
        ))

    _client.upsert(collection_name=settings.COLLECTION_NAME, points=points)


# ── Searching ─────────────────────────────────────────────────────────────────

def search(
    question: str,
    mode: str   = "hybrid",
    top_k: int  = 5,
    candidates: int = 20,
) -> list[dict]:
    """
    Embed the question and search Qdrant.

    mode="hybrid"  → dense + sparse branches fused with RRF
    mode="dense"   → dense only
    mode="sparse"  → sparse only

    Returns a list of chunk dicts, each with an added "score" field.
    """
    # Embed the query with both models
    dense_q  = emb.embed_dense([question])[0]
    sparse_q = emb.embed_sparse([question])[0]

    if mode == "hybrid":
        results = _hybrid_search(dense_q, sparse_q, top_k, candidates)
    elif mode == "dense":
        results = _dense_search(dense_q, top_k)
    elif mode == "sparse":
        results = _sparse_search(sparse_q, top_k)
    else:
        raise ValueError(f"Unknown SEARCH_MODE '{mode}'. Use: hybrid | dense | sparse")

    # Convert ScoredPoint objects to plain dicts
    chunks = []
    for r in results:
        chunk = dict(r.payload)
        chunk["score"] = r.score
        chunks.append(chunk)

    return chunks


def _hybrid_search(dense_q, sparse_q, top_k: int, candidates: int):
    """Dense + sparse prefetch, fused with Reciprocal Rank Fusion."""
    return _client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_q.tolist(),
                using="dense",
                limit=candidates,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist(),
                ),
                using="sparse",
                limit=candidates,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points


def _dense_search(dense_q, top_k: int):
    """Dense-only search."""
    return _client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=dense_q.tolist(),
        using="dense",
        limit=top_k,
    ).points


def _sparse_search(sparse_q, top_k: int):
    """Sparse-only (BM25) search."""
    return _client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=SparseVector(
            indices=sparse_q.indices.tolist(),
            values=sparse_q.values.tolist(),
        ),
        using="sparse",
        limit=top_k,
    ).points
