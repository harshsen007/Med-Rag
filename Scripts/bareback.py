#!/usr/bin/env python3
"""
bareback — Inspect intermediate inputs and outputs across all pipeline stages in clean, pure JSON.

Available commands:
    uv run bareback parsing <file_path>
    uv run bareback cleaning <file_path>
    uv run bareback chunking <file_path>
    uv run bareback embedding <text_or_file_path>
    uv run bareback ingest <file_path>
    uv run bareback retrieve [<file_path>] <query>

Examples:
    uv run bareback parsing Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback cleaning Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback chunking Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback embedding "Hemoptysis causes include pulmonary embolism"
    uv run bareback embedding Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback ingest Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback retrieve "What are the causes of hemoptysis?"
    uv run bareback retrieve Input_docs/clinical_rag_symptom_matrix.pdf "What are the causes of hemoptysis?"
"""

import sys
import os
import json
import contextlib
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@contextlib.contextmanager
def silence_stdout():
    """Silence stdout during internal library initialization so JSON output is pure."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ── 1. PARSING ───────────────────────────────────────────────────────────────

def run_parsing(file_path_str: str):
    from src.config import settings
    from src import parsing

    path = Path(file_path_str)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {file_path_str}"}, indent=2))
        sys.exit(1)

    with silence_stdout():
        raw_pages = parsing.parse_file(
            path,
            ocr_enabled=settings.OCR_ENABLED,
            ocr_language=settings.OCR_LANGUAGE,
        )

    output = {
        "command": "parsing",
        "input": {
            "file_path": str(path),
            "file_name": path.name,
            "format": path.suffix.lower(),
            "config": {
                "ocr_enabled": settings.OCR_ENABLED,
                "ocr_language": settings.OCR_LANGUAGE,
            },
        },
        "stats": {
            "total_pages": len(raw_pages),
            "total_characters": sum(len(txt) for _, txt in raw_pages),
        },
        "pages": [
            {
                "page": page_num,
                "char_length": len(text),
                "text": text,
            }
            for page_num, text in raw_pages
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── 2. CLEANING ──────────────────────────────────────────────────────────────

def run_cleaning(file_path_str: str):
    from src.config import settings
    from src import parsing, cleaning

    path = Path(file_path_str)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {file_path_str}"}, indent=2))
        sys.exit(1)

    with silence_stdout():
        raw_pages = parsing.parse_file(
            path,
            ocr_enabled=settings.OCR_ENABLED,
            ocr_language=settings.OCR_LANGUAGE,
        )
        cleaned_pages = cleaning.clean_pages(raw_pages)

    raw_map = {page_num: text for page_num, text in raw_pages}

    output = {
        "command": "cleaning",
        "input": {
            "file_path": str(path),
            "file_name": path.name,
            "raw_page_count": len(raw_pages),
        },
        "stats": {
            "raw_pages": len(raw_pages),
            "cleaned_pages": len(cleaned_pages),
            "raw_total_chars": sum(len(txt) for _, txt in raw_pages),
            "cleaned_total_chars": sum(len(txt) for _, txt in cleaned_pages),
        },
        "pages": [
            {
                "page": page_num,
                "raw_chars": len(raw_map.get(page_num, "")),
                "cleaned_chars": len(text),
                "chars_removed": len(raw_map.get(page_num, "")) - len(text),
                "cleaned_text": text,
            }
            for page_num, text in cleaned_pages
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── 3. CHUNKING ──────────────────────────────────────────────────────────────

def run_chunking(file_path_str: str):
    from src.config import settings
    from src import parsing, cleaning, chunking

    path = Path(file_path_str)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {file_path_str}"}, indent=2))
        sys.exit(1)

    with silence_stdout():
        raw_pages = parsing.parse_file(path, ocr_enabled=settings.OCR_ENABLED)
        cleaned_pages = cleaning.clean_pages(raw_pages)
        chunks = chunking.chunk_document(
            cleaned_pages,
            source=path.name,
            max_chars=settings.MAX_CHUNK_CHARS,
            min_chars=settings.MIN_CHUNK_CHARS,
            overlap_chars=settings.CHUNK_OVERLAP_CHARS,
        )

    output = {
        "command": "chunking",
        "input": {
            "file_path": str(path),
            "file_name": path.name,
            "config": {
                "max_chunk_chars": settings.MAX_CHUNK_CHARS,
                "min_chunk_chars": settings.MIN_CHUNK_CHARS,
                "chunk_overlap_chars": settings.CHUNK_OVERLAP_CHARS,
                "ocr_enabled": settings.OCR_ENABLED,
            },
        },
        "stats": {
            "raw_page_count": len(raw_pages),
            "cleaned_page_count": len(cleaned_pages),
            "total_chunks": len(chunks),
            "min_chunk_chars": min((len(c["text"]) for c in chunks), default=0),
            "max_chunk_chars": max((len(c["text"]) for c in chunks), default=0),
        },
        "chunks": chunks,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── 4. EMBEDDING ─────────────────────────────────────────────────────────────

def run_embedding(args: list[str]):
    from src.config import settings
    from src import embedding, parsing, cleaning, chunking

    if not args:
        print(json.dumps({"error": "Usage: bareback embedding <text_or_file_path>"}, indent=2))
        sys.exit(1)

    target = " ".join(args).strip()
    path = Path(target)

    # If it's an existing file, embed its first few chunks for demonstration
    if path.exists() and path.is_file():
        with silence_stdout():
            raw_pages = parsing.parse_file(path, ocr_enabled=settings.OCR_ENABLED)
            cleaned_pages = cleaning.clean_pages(raw_pages)
            chunks = chunking.chunk_document(
                cleaned_pages,
                source=path.name,
                max_chars=settings.MAX_CHUNK_CHARS,
                min_chars=settings.MIN_CHUNK_CHARS,
                overlap_chars=settings.CHUNK_OVERLAP_CHARS,
            )
            # Sample first 2 chunks if available to keep JSON payload manageable
            sample_chunks = chunks[:2] if chunks else []
            texts = [c["text"] for c in sample_chunks]
            dense_vecs = embedding.embed_dense(texts)
            sparse_vecs = embedding.embed_sparse(texts)

        items = []
        for i, (chunk, dv, sv) in enumerate(zip(sample_chunks, dense_vecs, sparse_vecs)):
            items.append({
                "chunk_index": chunk.get("chunk_index", i),
                "source": chunk.get("source"),
                "text_snippet": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
                "dense": {
                    "model": settings.DENSE_MODEL,
                    "dimension": len(dv),
                    "vector_sample_first_5": dv[:5].tolist(),
                },
                "sparse": {
                    "model": settings.SPARSE_MODEL,
                    "indices": sv.indices.tolist()[:10],
                    "values": [round(float(v), 4) for v in sv.values.tolist()[:10]],
                    "total_nonzero_terms": len(sv.indices),
                },
            })

        output = {
            "command": "embedding",
            "input": {
                "source_file": str(path),
                "total_chunks_in_file": len(chunks),
                "embedded_sample_count": len(sample_chunks),
            },
            "models": {
                "dense": settings.DENSE_MODEL,
                "sparse": settings.SPARSE_MODEL,
            },
            "embeddings": items,
        }
    else:
        # User provided an arbitrary text string or query
        query_text = target
        with silence_stdout():
            dv = embedding.embed_dense([query_text])[0]
            sv = embedding.embed_sparse([query_text])[0]

        output = {
            "command": "embedding",
            "input": {
                "text": query_text,
                "char_length": len(query_text),
            },
            "models": {
                "dense": settings.DENSE_MODEL,
                "sparse": settings.SPARSE_MODEL,
            },
            "dense": {
                "dimension": len(dv),
                "vector_sample_first_5": dv[:5].tolist(),
                "full_vector": dv.tolist(),
            },
            "sparse": {
                "indices": sv.indices.tolist(),
                "values": [round(float(v), 4) for v in sv.values.tolist()],
                "total_nonzero_terms": len(sv.indices),
            },
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── 5. INGEST ────────────────────────────────────────────────────────────────

def run_ingest(file_path_str: str):
    from src.config import settings
    from src import ingest, parsing, cleaning, chunking, vector_store

    path = Path(file_path_str)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {file_path_str}"}, indent=2))
        sys.exit(1)

    with silence_stdout():
        # Step 1: Parse
        pages = parsing.parse_file(
            path,
            ocr_enabled=settings.OCR_ENABLED,
            ocr_language=settings.OCR_LANGUAGE,
        )
        # Step 2: Clean
        pages = cleaning.clean_pages(pages)
        # Step 3: Chunk
        chunks = chunking.chunk_document(
            pages,
            source=path.name,
            max_chars=settings.MAX_CHUNK_CHARS,
            min_chars=settings.MIN_CHUNK_CHARS,
            overlap_chars=settings.CHUNK_OVERLAP_CHARS,
        )
        # Step 4: Ensure & Upsert in Qdrant
        vector_store.ensure_collection()
        vector_store.delete_by_source(path.name)
        if chunks:
            vector_store.upsert_chunks(chunks)

    output = {
        "command": "ingest",
        "input": {
            "file_path": str(path),
            "file_name": path.name,
        },
        "target_storage": {
            "qdrant_url": settings.QDRANT_URL,
            "collection_name": settings.COLLECTION_NAME,
        },
        "pipeline_summary": {
            "pages_parsed_and_cleaned": len(pages),
            "chunks_created": len(chunks),
            "points_upserted": len(chunks),
            "status": "success",
        },
        "sample_upserted_chunk": chunks[0] if chunks else None,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── 6. RETRIEVE ──────────────────────────────────────────────────────────────

def run_retrieve(args: list[str]):
    from src.config import settings
    from src import vector_store

    if not args:
        print(json.dumps({"error": "Usage: bareback retrieve [<file_path>] <query>"}, indent=2))
        sys.exit(1)

    source_filter = None
    first_arg = Path(args[0])

    if first_arg.exists() or first_arg.suffix.lower() in [".pdf", ".md", ".docx", ".html", ".htm", ".txt"]:
        source_filter = first_arg.name
        query = " ".join(args[1:]).strip()
        if not query:
            print(json.dumps({"error": f"Missing query after file path '{args[0]}'"}, indent=2))
            sys.exit(1)
    else:
        query = " ".join(args).strip()

    with silence_stdout():
        results = vector_store.search(
            question=query,
            mode=settings.SEARCH_MODE,
            top_k=settings.TOP_K,
            candidates=settings.CANDIDATES_PER_BRANCH,
        )

    if source_filter:
        results = [c for c in results if c.get("source") == source_filter]

    output = {
        "command": "retrieve",
        "input": {
            "query": query,
            "source_filter": source_filter,
            "search_mode": settings.SEARCH_MODE,
            "top_k": settings.TOP_K,
            "candidates_per_branch": settings.CANDIDATES_PER_BRANCH,
        },
        "stats": {
            "total_retrieved": len(results),
        },
        "results": results,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


# ── DISPATCHER ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "error": "Missing subcommand",
                    "usage": [
                        "bareback parsing <file_path>",
                        "bareback cleaning <file_path>",
                        "bareback chunking <file_path>",
                        "bareback embedding <text_or_file_path>",
                        "bareback ingest <file_path>",
                        "bareback retrieve [<file_path>] <query>",
                    ],
                },
                indent=2,
            )
        )
        sys.exit(1)

    cmd = sys.argv[1].lower()
    rest = sys.argv[2:]

    if cmd in ("parse", "parsing"):
        if not rest:
            print(json.dumps({"error": "Usage: bareback parsing <file_path>"}, indent=2))
            sys.exit(1)
        run_parsing(rest[0])
    elif cmd in ("clean", "cleaning"):
        if not rest:
            print(json.dumps({"error": "Usage: bareback cleaning <file_path>"}, indent=2))
            sys.exit(1)
        run_cleaning(rest[0])
    elif cmd in ("chunk", "chunking"):
        if not rest:
            print(json.dumps({"error": "Usage: bareback chunking <file_path>"}, indent=2))
            sys.exit(1)
        run_chunking(rest[0])
    elif cmd in ("embed", "embedding"):
        run_embedding(rest)
    elif cmd == "ingest":
        if not rest:
            print(json.dumps({"error": "Usage: bareback ingest <file_path>"}, indent=2))
            sys.exit(1)
        run_ingest(rest[0])
    elif cmd in ("retrieve", "retreive"):
        run_retrieve(rest)
    else:
        print(
            json.dumps(
                {
                    "error": f"Unknown subcommand '{cmd}'",
                    "valid_commands": [
                        "parsing",
                        "cleaning",
                        "chunking",
                        "embedding",
                        "ingest",
                        "retrieve",
                    ],
                },
                indent=2,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
