# MedRAG — Agents & Architecture

> How the pipeline is structured and how each piece fits together.

---

## Overview

MedRAG is a Retrieval-Augmented Generation system for medical documents.
You point it at PDFs, Markdown files, DOCX, or HTML — it parses them, stores them in Qdrant, and then answers questions with cited sources.

```
INGEST
  file (pdf / md / docx / html)
    → parsing.py     convert to Markdown (+ optional OCR for scanned PDFs)
    → cleaning.py    fix character artefacts, remove headers/footers
    → chunking.py    split into context-aware chunks with heading metadata
    → embedding.py   dense vector + sparse BM25 vector per chunk
    → vector_store.py  upsert into Qdrant (two named vectors per point)

QUERY
  question
    → embedding.py   embed the question (dense + sparse)
    → vector_store.py  hybrid search → top-k chunks
    → generate.py    fill prompt with chunks → LLM → cited answer
```

---

## Agents / Modules

Each file has **one responsibility**. There are no abstract base classes, no inheritance chains.

| File | Responsibility |
|---|---|
| `src/config.py` | Single `Settings` object (pydantic-settings). All tunable values live here. Secrets come from `.env`. |
| `src/prompts.py` | `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`. Not config — edit when the prompt logic changes. |
| `src/parsing.py` | Detect file format → dispatch to the right parser → return `list[(page, markdown)]`. |
| `src/cleaning.py` | Fix broken characters, remove repeated headers/footers, normalise whitespace. |
| `src/chunking.py` | Extract heading-based sections → pack blocks into chunks → apply overlap and merge small chunks. |
| `src/embedding.py` | Two lazy singletons (dense + sparse fastembed models). `embed_dense()` and `embed_sparse()`. |
| `src/vector_store.py` | Create/manage the Qdrant collection. `upsert_chunks()` and `search()`. |
| `src/ingest.py` | Orchestrate parse → clean → chunk → embed → store for one file or a whole directory. |
| `src/retrieve.py` | `retrieve(question)` → embed + search → list of chunk dicts. |
| `src/generate.py` | `answer(question, chunks)` → format chunks into prompt → call LLM → return text. |
| `src/main.py` | CLI entry point: `ingest`, `inspect`, `ask`. |
| `eval.py` | Run Ragas evaluation over all three search modes. |

---

## Data flow in detail

### Ingestion

```
parse_file(path)
  └─ returns [(1, "## Heading\n..."), (2, "..."), ...]

clean_pages(pages)
  └─ fixes \x00 → "fi", private chars → "-"
  └─ drops bare page numbers and repeated header/footer lines
  └─ collapses multiple blank lines

chunk_document(pages, source)
  └─ _extract_blocks() — walks lines, groups into para / list / table blocks per section
  └─ _pack_blocks()    — greedy packing up to MAX_CHUNK_CHARS with overlap
  └─ _merge_small()    — merge chunks < MIN_CHUNK_CHARS into neighbours
  └─ returns list of chunk dicts

upsert_chunks(chunks)
  └─ embed_dense(texts) + embed_sparse(texts)
  └─ builds PointStruct with dense vector + SparseVector
  └─ _client.upsert(collection_name, points)
```

### Query

```
retrieve(question)
  └─ embed_dense([question])[0] + embed_sparse([question])[0]
  └─ search(mode="hybrid", top_k=5, candidates=20)
       └─ _hybrid_search: Prefetch(dense) + Prefetch(sparse) → FusionQuery(RRF)
  └─ returns list of chunk dicts with "score" added

answer(question, chunks)
  └─ formats chunks as numbered list with source / section / page labels
  └─ fills USER_PROMPT_TEMPLATE
  └─ OpenAI(api_key=NVIDIA_API_KEY, base_url=LLM_BASE_URL)
       .chat.completions.create(...)
  └─ returns response text
```

---

## Qdrant collection schema

```
Collection: medical_docs
  vectors:
    dense  → VectorParams(size=384, distance=COSINE)
    sparse → SparseVectorParams(modifier=IDF)   ← required for Qdrant/bm25

  Point payload (= all chunk fields):
    text        str
    source      str   (filename)
    doc_type    str   (pdf | md | docx | html)
    section     str   (heading path, e.g. "Respiratory > Hemoptysis")
    page_start  int | None
    page_end    int | None
    chunk_index int

  Point ID: int  (MD5 hash of "source::chunk_index", truncated to 63 bits)
  Reason: stable IDs mean re-ingesting a file overwrites the old points
          without touching anything from other files.
```

---

## Search modes

| Mode | What happens |
|---|---|
| `hybrid` | Dense branch + sparse branch, fused with Reciprocal Rank Fusion (RRF). Best overall. |
| `dense`  | Semantic similarity only. Good for conceptual questions. |
| `sparse` | BM25 keyword match only. Good for exact drug names, codes, abbreviations. |

Switch with `SEARCH_MODE=dense` in `.env` or `config.py`.

---

## CLI quick reference

```bash
# Ingest all files in Input_docs/
uv run python -m src.main ingest

# Ingest one specific file
uv run python -m src.main ingest Input_docs/clinical_medicine.pdf

# Check chunking without writing to Qdrant
uv run python -m src.main inspect Input_docs/clinical_medicine.pdf

# Ask a question
uv run python -m src.main ask "What are the five causes of hemoptysis?"

# Run Ragas eval (all three modes)
uv run python eval.py
```

---

## Not in v1 (intentionally excluded)

- Reranker
- Parent-child retrieval
- Query rewriting
- Image captioning / vision model
- Web UI
- Agents
- Caching
- PubMed / live data pipeline (separate future work)
