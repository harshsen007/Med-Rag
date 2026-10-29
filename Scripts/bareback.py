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
    uv run bareback generate [--stream] [<file_path>] <query>

Examples:
    uv run bareback parsing Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback cleaning Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback chunking Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback embedding "Hemoptysis causes include pulmonary embolism"
    uv run bareback embedding Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback ingest Input_docs/clinical_rag_symptom_matrix.pdf
    uv run bareback retrieve "What are the causes of hemoptysis?"
    uv run bareback retrieve Input_docs/clinical_rag_symptom_matrix.pdf "What are the causes of hemoptysis?"
    uv run bareback generate "What are the causes of hemoptysis?"
    uv run bareback generate --stream "What are the causes of hemoptysis?"
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


# ── 7. GENERATE (LLM) ─────────────────────────────────────────────────────────

def run_generate(args: list[str]):
    from openai import OpenAI
    from src.config import settings
    from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from src import vector_store

    if not args:
        print(
            json.dumps(
                {
                    "error": "Usage: bareback generate [--stream] [<file_path>] <query>",
                    "examples": [
                        'bareback generate "What are the causes of hemoptysis?"',
                        'bareback generate --stream "What are the causes of hemoptysis?"',
                        'bareback generate Input_docs/clinical_rag_symptom_matrix.pdf "What are the causes of hemoptysis?"',
                    ],
                },
                indent=2,
            )
        )
        sys.exit(1)

    # Check for stream flag
    stream_mode = False
    clean_args = []
    for arg in args:
        if arg in ("--stream", "-s", "--streaming"):
            stream_mode = True
        else:
            clean_args.append(arg)

    if not clean_args:
        print(json.dumps({"error": "Missing query for generate command."}, indent=2))
        sys.exit(1)

    # Check if first positional arg is a source file path
    source_filter = None
    first_arg = Path(clean_args[0])
    if first_arg.exists() or first_arg.suffix.lower() in [".pdf", ".md", ".docx", ".html", ".htm", ".txt"]:
        source_filter = first_arg.name
        query = " ".join(clean_args[1:]).strip()
        if not query:
            print(json.dumps({"error": f"Missing query after file path '{clean_args[0]}'"}, indent=2))
            sys.exit(1)
    else:
        query = " ".join(clean_args).strip()

    # Step 1: Retrieve context chunks
    with silence_stdout():
        results = vector_store.search(
            question=query,
            mode=settings.SEARCH_MODE,
            top_k=settings.TOP_K,
            candidates=settings.CANDIDATES_PER_BRANCH,
        )

    if source_filter:
        results = [c for c in results if c.get("source") == source_filter]

    # Step 2: Format chunks for prompt
    chunk_texts = []
    for i, chunk in enumerate(results, start=1):
        source = chunk.get("source", "unknown")
        section = chunk.get("section", "")
        pg_s = chunk.get("page_start")
        pg_e = chunk.get("page_end")

        if pg_s and pg_e and pg_s != pg_e:
            page_label = f"p.{pg_s}-{pg_e}"
        elif pg_s:
            page_label = f"p.{pg_s}"
        else:
            page_label = "n/a"

        header = f"[{i}] Source: {source} | Section: {section} | Page: {page_label}"
        chunk_texts.append(f"{header}\n{chunk['text']}")

    formatted_chunks = "\n\n---\n\n".join(chunk_texts)

    # Step 3: Construct prompts
    user_prompt = USER_PROMPT_TEMPLATE.format(
        chunks=formatted_chunks,
        question=query,
    )

    messages_payload = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    llm_call_config = {
        "base_url": settings.LLM_BASE_URL,
        "model": settings.LLM_MODEL,
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "stream": stream_mode,
        "api_key_configured": bool(settings.NVIDIA_API_KEY),
    }

    client = OpenAI(
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )

    if not stream_mode:
        # Non-streaming LLM invocation
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages_payload,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

        raw_response = response.model_dump()
        answer_text = response.choices[0].message.content if response.choices else ""
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)

        output = {
            "command": "generate",
            "input": {
                "query": query,
                "source_filter": source_filter,
                "retrieved_chunks_count": len(results),
            },
            "llm_call": llm_call_config,
            "prompts": {
                "system_prompt": SYSTEM_PROMPT,
                "formatted_context_passages": formatted_chunks,
                "user_prompt": user_prompt,
                "messages_payload": messages_payload,
            },
            "raw_response": raw_response,
            "reasoning_content": reasoning_content,
            "answer": answer_text,
            "retrieved_chunks": results,
        }
    else:
        # Streaming LLM invocation: deltas are streamed live to stderr for real-time viewing
        sys.stderr.write(f"\n[bareback stream connecting to {settings.LLM_MODEL}...]\n")
        sys.stderr.flush()

        stream = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages_payload,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            stream=True,
        )

        streamed_chunks = []
        content_deltas = []
        reasoning_deltas = []

        for chunk in stream:
            chunk_dict = chunk.model_dump()
            streamed_chunks.append(chunk_dict)

            if chunk.choices:
                delta = chunk.choices[0].delta
                r = getattr(delta, "reasoning_content", None)
                c = getattr(delta, "content", None)
                if r:
                    reasoning_deltas.append(r)
                    sys.stderr.write(r)
                    sys.stderr.flush()
                if c:
                    content_deltas.append(c)
                    sys.stderr.write(c)
                    sys.stderr.flush()

        sys.stderr.write("\n[stream complete]\n\n")
        sys.stderr.flush()

        raw_response_sample = {
            "stream_chunk_count": len(streamed_chunks),
            "first_chunk": streamed_chunks[0] if streamed_chunks else None,
            "last_chunk": streamed_chunks[-1] if streamed_chunks else None,
        }

        assembled_answer = "".join(content_deltas)
        assembled_reasoning = "".join(reasoning_deltas)

        output = {
            "command": "generate",
            "input": {
                "query": query,
                "source_filter": source_filter,
                "retrieved_chunks_count": len(results),
            },
            "llm_call": llm_call_config,
            "prompts": {
                "system_prompt": SYSTEM_PROMPT,
                "formatted_context_passages": formatted_chunks,
                "user_prompt": user_prompt,
                "messages_payload": messages_payload,
            },
            "raw_response": raw_response_sample,
            "stream_details": {
                "total_streamed_chunks": len(streamed_chunks),
                "reasoning_chunks_count": len(reasoning_deltas),
                "content_chunks_count": len(content_deltas),
                "content_deltas": content_deltas,
            },
            "reasoning_content": assembled_reasoning or None,
            "answer": assembled_answer,
            "retrieved_chunks": results,
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
                        "bareback generate [--stream] [<file_path>] <query>",
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
    elif cmd in ("generate", "ask", "llm"):
        run_generate(rest)
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
                        "generate",
                    ],
                },
                indent=2,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
