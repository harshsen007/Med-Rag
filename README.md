# MedRAG — Clinical Medical Retrieval-Augmented Generation

MedRAG is an end-to-end, deterministic Retrieval-Augmented Generation (RAG) system specialized for medical and clinical documents (PDF, Markdown, DOCX, HTML). It parses documents, cleans OCR/formatting artefacts, splits text with heading-aware context chunking, computes dense (`BAAI/bge-small-en-v1.5`) and sparse BM25 (`Qdrant/bm25`) embeddings, and stores them in **Qdrant** for hybrid search fusion with cited LLM generation.

---

## 🏗️ Architecture & Pipeline Flow

```
                      ┌────────────────────────────┐
                      │ Input Documents            │
                      │ (PDF, Markdown, DOCX, HTML)│
                      └─────────────┬──────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. PARSING (src/parsing.py)                                           │
│    • PDF (PyMuPDF / pymupdf4llm) + Tesseract OCR for scanned pages     │
│    • Markdown / DOCX / HTML (MarkItDown)                               │
│    • Output: list[ (page_number, markdown_text) ]                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. CLEANING (src/cleaning.py)                                         │
│    • Fixes ligatures (\x00 -> "fi", private unicode -> "-")            │
│    • Removes bare page numbers and repeated header/footer lines        │
│    • Normalizes whitespace and drops empty pages                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. CONTEXT-AWARE CHUNKING (src/chunking.py)                           │
│    • Tracks markdown heading hierarchy ("Cardiology > Arrhythmias")    │
│    • Packs semantic blocks (paragraphs, lists, tables)                │
│    • Applies overlap and merges chunks < MIN_CHUNK_CHARS               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. EMBEDDING & STORAGE (src/embedding.py & src/vector_store.py)        │
│    • Dense Vector: BAAI/bge-small-en-v1.5 (384-dim, Cosine)            │
│    • Sparse BM25:  Qdrant/bm25 (with IDF modifier)                     │
│    • Qdrant Points: deterministic hash ID (`source::chunk_index`)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. RETRIEVAL & ANSWER GENERATION (src/retrieve.py & src/generate.py)   │
│    • Hybrid Search: Dense + Sparse branches fused via RRF              │
│    • Top-k retrieved chunks injected into strict clinical prompt       │
│    • LLM (NVIDIA Nemotron / OpenAI compatible) generates cited answer  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Requirements & Prerequisites
- Python `>= 3.12, < 3.14`
- [uv](https://github.com/astral-sh/uv) (fast Python package and project manager)
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose (for Qdrant vector database)

### 2. Environment Setup
Create a `.env` file in the project root:
```bash
cp .env.example .env   # or create .env directly
```
Add your configurations (defaults in `src/config.py`):
```env
# Vector Database
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=medical_docs

# Search & Retrieval Configuration
SEARCH_MODE=hybrid      # Options: hybrid | dense | sparse
TOP_K=5
CANDIDATES_PER_BRANCH=20

# NVIDIA LLM Configuration
NVIDIA_API_KEY=nvapi-your-key-here
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
```

---

## 🐳 Starting Qdrant with Docker

Run Qdrant using the provided `docker-compose.yml`:

```bash
# Start Qdrant in the background
docker compose up -d
```

Or using standalone Docker:
```bash
docker run -d --name medrag_qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

- **REST API & Dashboard:** Visit [http://localhost:6333/dashboard](http://localhost:6333/dashboard) to view collections and vectors.
- **Stop Qdrant:** `docker compose down`

---

## 📖 CLI Usage (`src.main`)

### 1. Ingest Documents into Qdrant
Ingest all files in `Input_docs/` or a single file:
```bash
# Ingest all files inside Input_docs/
uv run python -m src.main ingest

# Ingest a specific document
uv run python -m src.main ingest Input_docs/clinical_rag_symptom_matrix.pdf
```
*Note: Re-ingesting a document safely overwrites its old chunks without altering points from other files.*

### 2. Inspect Chunk Statistics (Dry-run)
Check how your chunker splits a document without writing anything to Qdrant:
```bash
uv run python -m src.main inspect Input_docs/clinical_rag_symptom_matrix.pdf
```

### 3. Ask a Clinical Question
Query your indexed documents and generate an answer with grounded citations:
```bash
uv run python -m src.main ask "What are the common presenting symptoms and causes of Pulmonary Embolism?"
```

---

## 🔍 Diagnostic & Inspection Scripts (`Scripts/`)

### Interactive Pipeline Walkthrough (`demo_pipeline.py`)
Prints stage-by-stage visual output with image counts and text previews:
```bash
uv run python Scripts/demo_pipeline.py
```

### Raw JSON Pipeline Inspector (`bareback.py`)
Inspect inputs, intermediate structures, and outputs at every step in clean, pipeable JSON.

```bash
# 1. Inspect Parsing (PDF text extraction, page counts)
uv run python Scripts/bareback.py parsing Input_docs/clinical_rag_symptom_matrix.pdf

# 2. Inspect Cleaning (characters dropped, header/footer strip)
uv run python Scripts/bareback.py cleaning Input_docs/clinical_rag_symptom_matrix.pdf

# 3. Inspect Chunking (semantic blocks, headings, page spans)
uv run python Scripts/bareback.py chunking Input_docs/clinical_rag_symptom_matrix.pdf

# 4. Inspect Embedding (dense dimensions & sparse BM25 non-zero terms)
uv run python Scripts/bareback.py embedding "What are the causes of hemoptysis?"

# 5. Inspect Full Ingestion (upserts points to Qdrant)
uv run python Scripts/bareback.py ingest Input_docs/clinical_rag_symptom_matrix.pdf

# 6. Inspect Retrieval (hybrid similarity search results & scores)
uv run python Scripts/bareback.py retrieve "What are the causes of hemoptysis?"
uv run python Scripts/bareback.py retrieve Input_docs/clinical_rag_symptom_matrix.pdf "What are the causes of hemoptysis?"

# 7. Inspect LLM Generation & Prompts (prompts, raw API response, token usage)
uv run python Scripts/bareback.py generate "What are the causes of hemoptysis?"
uv run python Scripts/bareback.py generate Input_docs/clinical_rag_symptom_matrix.pdf "What are the causes of hemoptysis?"

# 8. Inspect Live LLM Streaming (streams tokens in real-time to stderr, outputs JSON to stdout)
uv run python Scripts/bareback.py generate --stream "What are the causes of hemoptysis?"
```

#### What `bareback generate` Exposes:
- **`llm_call`**: API endpoint (`base_url`), `model`, `temperature`, `max_tokens`, and stream configuration.
- **`prompts`**: Exact clinical `system_prompt`, formatted passage block with source/page citations, and the interpolated `user_prompt`.
- **`raw_response`**: Verbatim raw response object from the NVIDIA/OpenAI API (`id`, `usage`: prompt/completion/total tokens, `finish_reason`, and model metadata).
- **`reasoning_content`**: Internal chain-of-thought tokens (when using reasoning models like Nemotron).
- **`stream_details`**: Streaming chunk counts and token breakdown when `--stream` is enabled.
- **`answer`**: Final grounded clinical answer text.

---

## 📊 RAG Evaluation (`eval.py`)

Run automated Ragas evaluation across all three search modes (`hybrid`, `dense`, `sparse`):
```bash
uv run python eval.py
```

---

## 📂 Project Structure

```
MedRag/
├── docker-compose.yml       # Qdrant service configuration
├── pyproject.toml           # Project dependencies and environment specs
├── .env                     # API keys and environment overrides
├── Input_docs/              # Reference clinical PDFs and source documents
├── Scripts/
│   ├── demo_pipeline.py     # Interactive terminal pipeline demonstration
│   └── bareback.py          # Pure JSON inspection CLI for all pipeline stages
├── src/
│   ├── config.py            # Pydantic Settings (tunable parameters)
│   ├── parsing.py           # Format detection + PDF/Markdown extraction + OCR
│   ├── cleaning.py          # Character correction and header/footer cleanup
│   ├── chunking.py          # Context-aware markdown chunking & packing
│   ├── embedding.py         # FastEmbed dense & sparse BM25 models
│   ├── vector_store.py      # Qdrant collection setup, upsert, and search
│   ├── ingest.py            # Ingestion pipeline orchestration
│   ├── retrieve.py          # Question retrieval interface
│   ├── generate.py          # Prompt builder and LLM client call
│   ├── prompts.py           # System & user prompt templates
│   └── main.py              # CLI entry point (ingest, inspect, ask)
└── eval.py                  # Ragas evaluation benchmark
```
