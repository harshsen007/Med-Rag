"""
src/main.py — Command-line interface.

Usage:
  python -m src.main ingest [path]          Ingest one file or all files in DOCS_DIR
  python -m src.main inspect <file>         Print chunk stats without touching Qdrant
  python -m src.main ask "your question"    Retrieve + generate an answer

Examples:
  python -m src.main ingest
  python -m src.main ingest Input_docs/clinical_medicine.pdf
  python -m src.main inspect Input_docs/clinical_medicine.pdf
  python -m src.main ask "What are the symptoms of pneumonia?"
"""

import sys
import statistics
from pathlib import Path

from src.config import settings
from src import ingest, retrieve, generate, parsing, cleaning, chunking


def cmd_ingest(args: list[str]):
    """Ingest one file or all files in DOCS_DIR."""
    if args:
        path = Path(args[0])
        if not path.exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)
        ingest.ingest_file(path)
    else:
        ingest.ingest_dir(settings.DOCS_DIR)


def cmd_inspect(args: list[str]):
    """
    Show chunk statistics for a file without writing anything to Qdrant.
    Useful for checking chunker settings before ingesting.
    """
    if not args:
        print("Usage: python -m src.main inspect <file>")
        sys.exit(1)

    path = Path(args[0])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    print(f"\nInspecting: {path.name}")

    # Run parse → clean → chunk (no embedding, no Qdrant)
    pages  = parsing.parse_file(path, ocr_enabled=settings.OCR_ENABLED)
    pages  = cleaning.clean_pages(pages)
    chunks = chunking.chunk_document(
        pages,
        source=path.name,
        max_chars=settings.MAX_CHUNK_CHARS,
        min_chars=settings.MIN_CHUNK_CHARS,
        overlap_chars=settings.CHUNK_OVERLAP_CHARS,
    )

    sizes = [len(c["text"]) for c in chunks]

    print(f"\n── Chunk statistics ─────────────────────")
    print(f"  Total chunks : {len(chunks)}")
    if sizes:
        print(f"  Min  size    : {min(sizes)} chars")
        print(f"  Median size  : {int(statistics.median(sizes))} chars")
        print(f"  Max  size    : {max(sizes)} chars")
    print()

    # Show the first 3 chunks
    preview_count = min(3, len(chunks))
    print(f"── First {preview_count} chunk(s) ──────────────────────")
    for i in range(preview_count):
        c = chunks[i]
        print(f"\n[Chunk {c['chunk_index']}]")
        print(f"  Section : {c['section'] or '(no heading)'}")
        print(f"  Pages   : {c['page_start']} – {c['page_end']}")
        print(f"  Size    : {len(c['text'])} chars")
        print(f"  Text preview:\n{c['text'][:300]}")
        print()


def cmd_ask(args: list[str]):
    """Retrieve chunks and generate an answer."""
    if not args:
        print('Usage: python -m src.main ask "your question"')
        sys.exit(1)

    question = " ".join(args)
    print(f"\nQuestion: {question}\n")

    # Retrieve
    chunks = retrieve.retrieve(question)

    print(f"── Retrieved {len(chunks)} chunk(s) ────────────────────")
    for i, c in enumerate(chunks, start=1):
        pg_s = c.get("page_start")
        pg_e = c.get("page_end")
        page_label = f"p.{pg_s}" if pg_s == pg_e else f"p.{pg_s}-{pg_e}" if pg_s else "n/a"
        print(f"  [{i}] {c.get('source','?')} | {c.get('section','?')} | {page_label} | score={c.get('score', 0):.4f}")
    print()

    # Generate
    answer_text = generate.answer(question, chunks=chunks)
    print("── Answer ───────────────────────────────")
    print(answer_text)


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "ingest":  cmd_ingest,
    "inspect": cmd_inspect,
    "ask":     cmd_ask,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    rest    = sys.argv[2:]
    COMMANDS[command](rest)
