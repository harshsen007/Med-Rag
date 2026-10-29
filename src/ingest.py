"""
src/ingest.py — Orchestrate the full ingestion pipeline for one file.

Pipeline:
  parse → clean → chunk → embed+store in Qdrant

Public function:
  ingest_file(path)  — ingest one file (re-ingest deletes old points first)
  ingest_dir(dir)    — ingest every supported file in a directory
"""

from pathlib import Path
from src.config import settings
from src import parsing, cleaning, chunking, vector_store

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".docx", ".html", ".htm"}


def ingest_file(path: Path):
    """
    Ingest one file into Qdrant.

    Steps:
      1. Parse → list of (page, markdown) pairs
      2. Clean → fix artefacts, remove headers/footers
      3. Chunk → list of chunk dicts
      4. Delete any existing points for this file
      5. Upsert new points
    """
    path = Path(path)
    print(f"\n── Ingesting: {path.name}")

    # 1. Parse
    pages = parsing.parse_file(
        path,
        ocr_enabled=settings.OCR_ENABLED,
        ocr_language=settings.OCR_LANGUAGE,
    )
    print(f"   Parsed  {len(pages)} page(s)")

    # 2. Clean
    pages = cleaning.clean_pages(pages)
    print(f"   After cleaning: {len(pages)} non-empty page(s)")

    # 3. Chunk
    chunks = chunking.chunk_document(
        pages,
        source=path.name,
        max_chars=settings.MAX_CHUNK_CHARS,
        min_chars=settings.MIN_CHUNK_CHARS,
        overlap_chars=settings.CHUNK_OVERLAP_CHARS,
    )
    print(f"   Created {len(chunks)} chunk(s)")

    if not chunks:
        print("   No chunks produced — skipping.")
        return

    # 4. Delete old points for this file
    vector_store.ensure_collection()
    vector_store.delete_by_source(path.name)
    print(f"   Deleted old points for '{path.name}'")

    # 5. Upsert new points
    vector_store.upsert_chunks(chunks)
    print(f"   Upserted {len(chunks)} point(s) into '{settings.COLLECTION_NAME}'")


def ingest_dir(directory: Path):
    """Ingest every supported file in a directory (non-recursive)."""
    directory = Path(directory)
    files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"No supported files found in {directory}")
        return

    print(f"Found {len(files)} file(s) in {directory}")
    for f in files:
        ingest_file(f)
