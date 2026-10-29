"""
src/chunking.py — Split cleaned Markdown pages into chunks ready for embedding.

How it works (step by step):
  1. Walk all pages line by line.  Every Markdown heading starts a new section
     and updates the heading path (e.g. "Respiratory > Hemoptysis").
  2. Inside each section, group lines into logical blocks:
       - A paragraph  = consecutive non-empty, non-heading, non-list lines.
       - A list block = consecutive bullet / numbered lines.
       - A table block = consecutive Markdown table rows.
  3. Pack blocks one by one into a chunk until adding the next block would
     exceed MAX_CHUNK_CHARS.  Then emit the current chunk and start a new one.
  4. If a single block is larger than MAX_CHUNK_CHARS, split it:
       - Tables  → split by data row (repeat the header in each piece).
       - Text    → split by sentence; hard-cut only as a last resort.
  5. When emitting a new chunk inside the same section, carry back the last
     block(s) whose total size fits in CHUNK_OVERLAP_CHARS (overlap).
     No overlap across section boundaries.
  6. After all sections are processed, merge any chunk shorter than
     MIN_CHUNK_CHARS into its neighbour (next if same section, else prev).
  7. Prepend "Section: <heading path>" to every chunk's text.

Each chunk dict has:
    text        str          (header line + body — this is what gets embedded)
    source      str          (file name)
    doc_type    str          (pdf | md | docx | html)
    section     str          (heading path)
    page_start  int | None
    page_end    int | None
    chunk_index int
"""

import re
from pathlib import Path


# ── Regex helpers ─────────────────────────────────────────────────────────────

_HEADING     = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_ROW   = re.compile(r"^\|.+\|$")
_LIST_ITEM   = re.compile(r"^\s*([-*+]|\d+\.)\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def _doc_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".pdf": "pdf", ".md": "md", ".markdown": "md",
            ".docx": "docx", ".html": "html", ".htm": "html"}.get(ext, "unknown")


# ── Step 1 & 2 — extract blocks from pages ───────────────────────────────────

def _extract_blocks(pages: list[tuple]) -> list[dict]:
    """
    Walk pages and return a flat list of block dicts.
    Each block has: text, page, kind ("para"|"table"|"list"), section, heading_stack.

    heading_stack is a list of (level, title) tuples that lets us rebuild the
    heading path later.
    """
    blocks = []
    heading_stack = []  # list of (level, title)

    # Accumulator state for the current block being built
    buf_kind  = None   # "para" | "table" | "list"
    buf_lines = []
    buf_page  = None

    def flush():
        """Emit whatever is in the buffer as a block."""
        nonlocal buf_kind, buf_lines, buf_page
        text = "\n".join(buf_lines).strip()
        if text and buf_kind:
            blocks.append({
                "text":    text,
                "page":    buf_page,
                "kind":    buf_kind,
                "section": _heading_path(heading_stack),
            })
        buf_kind  = None
        buf_lines = []
        buf_page  = None

    def start(kind, line, page):
        nonlocal buf_kind, buf_lines, buf_page
        flush()
        buf_kind  = kind
        buf_lines = [line]
        buf_page  = page

    for page_num, text in pages:
        for line in text.split("\n"):

            # ── Heading ──────────────────────────────────────────────────────
            m = _HEADING.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                # Keep only headings above this level
                heading_stack = [(l, t) for l, t in heading_stack if l < level]
                heading_stack.append((level, title))
                continue

            # ── Table row ────────────────────────────────────────────────────
            if _TABLE_ROW.match(line.strip()):
                if buf_kind != "table":
                    start("table", line, page_num)
                else:
                    buf_lines.append(line)
                continue

            # If we were in a table and this line is not a table row, end it
            if buf_kind == "table":
                flush()

            # ── List item ────────────────────────────────────────────────────
            if _LIST_ITEM.match(line):
                if buf_kind != "list":
                    start("list", line, page_num)
                else:
                    buf_lines.append(line)
                continue

            # ── Blank line → end of paragraph / list ─────────────────────────
            if not line.strip():
                if buf_kind in {"para", "list"}:
                    flush()
                continue

            # ── Regular paragraph line ────────────────────────────────────────
            if buf_kind != "para":
                start("para", line, page_num)
            else:
                buf_lines.append(line)

    flush()
    return blocks


def _heading_path(stack: list[tuple]) -> str:
    return " > ".join(title for _, title in stack)


# ── Step 3–5 — pack blocks into chunks ───────────────────────────────────────

def _pack_blocks(blocks: list[dict], source: str, max_chars: int, overlap_chars: int) -> list[dict]:
    """
    Greedy packing: add blocks to the current chunk until it would exceed
    max_chars, then emit and start a new chunk with optional overlap.
    """
    chunks = []
    current_blocks = []
    current_chars  = 0
    current_section = None
    page_start = None
    page_end   = None

    def emit():
        """Build a chunk dict from current_blocks and append to chunks."""
        body = "\n\n".join(b["text"] for b in current_blocks)
        section = current_blocks[-1]["section"] if current_blocks else ""
        header = f"Section: {section}" if section else "Section: (document start)"
        chunks.append({
            "text":        f"{header}\n\n{body}",
            "source":      source,
            "doc_type":    _doc_type(source),
            "section":     section,
            "page_start":  page_start,
            "page_end":    page_end,
            "chunk_index": len(chunks),
        })

    for blk in blocks:
        # Split oversized blocks before packing
        pieces = _split_block(blk, max_chars)

        for piece in pieces:
            size = len(piece["text"])

            # Section boundary → emit without overlap
            if piece["section"] != current_section and current_blocks:
                emit()
                current_blocks = []
                current_chars  = 0
                page_start = None
                page_end   = None

            current_section = piece["section"]

            # Would adding this piece exceed the limit?
            if current_chars + size > max_chars and current_blocks:
                emit()

                # Overlap: carry trailing blocks that fit within overlap_chars
                carry = []
                carry_chars = 0
                for b in reversed(current_blocks):
                    if carry_chars + len(b["text"]) <= overlap_chars:
                        carry.insert(0, b)
                        carry_chars += len(b["text"])
                    else:
                        break
                current_blocks = carry
                current_chars  = carry_chars
                page_start = carry[0]["page"] if carry else None
                page_end   = carry[-1]["page"] if carry else None

            if page_start is None and piece["page"] is not None:
                page_start = piece["page"]
            if piece["page"] is not None:
                page_end = piece["page"]

            current_blocks.append(piece)
            current_chars += size

    if current_blocks:
        emit()

    return chunks


# ── Splitting oversized blocks ────────────────────────────────────────────────

def _split_block(blk: dict, max_chars: int) -> list[dict]:
    """If a block is too large, split it.  Returns a list of smaller block dicts."""
    if len(blk["text"]) <= max_chars:
        return [blk]

    if blk["kind"] == "table":
        texts = _split_table(blk["text"], max_chars)
    else:
        texts = _split_text(blk["text"], max_chars)

    return [{**blk, "text": t} for t in texts]


def _split_table(text: str, max_chars: int) -> list[str]:
    """Split a Markdown table by data rows, repeating the header in each piece."""
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return [text]

    header = lines[0] + "\n" + lines[1] + "\n"   # column names + separator
    pieces = []
    current = header

    for row in lines[2:]:
        if len(current) + len(row) > max_chars and current != header:
            pieces.append(current.strip())
            current = header + row + "\n"
        else:
            current += row + "\n"

    if current.strip() and current.strip() != header.strip():
        pieces.append(current.strip())
    return pieces or [text]


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text by sentence boundaries; hard-cut only if a sentence is too long."""
    sentences = _SENTENCE_END.split(text)
    pieces = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = sent
        else:
            current = candidate

    if current:
        pieces.append(current)

    # Last resort: hard cut any piece that is still too long
    final = []
    for p in pieces:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))

    return final or [text]


# ── Step 6 — merge small chunks ───────────────────────────────────────────────

def _merge_small(chunks: list[dict], min_chars: int) -> list[dict]:
    """
    Merge any chunk shorter than min_chars into its neighbour.
    Prefer merging forward (into the next chunk); merge backward only for the
    last chunk.  Re-index chunk_index after merging.
    """
    result = []
    carry  = None  # a small chunk waiting to be merged

    for chunk in chunks:
        if carry is not None:
            # Merge carry into this chunk
            carry_body = carry["text"].split("\n\n", 1)[-1]
            this_body  = chunk["text"].split("\n\n", 1)[-1]
            section    = chunk["section"]
            header     = f"Section: {section}" if section else "Section: (document start)"
            merged_text = f"{header}\n\n{carry_body}\n\n{this_body}"
            chunk = {
                **chunk,
                "text":       merged_text,
                "page_start": carry["page_start"] or chunk["page_start"],
                "page_end":   chunk["page_end"]   or carry["page_end"],
            }
            carry = None

        if len(chunk["text"]) < min_chars:
            carry = chunk  # hold it; try to merge into the next chunk
        else:
            result.append(chunk)

    # If the very last chunk is small, merge it backwards
    if carry is not None:
        if result:
            last = result[-1]
            last_body  = last["text"].split("\n\n", 1)[-1]
            carry_body = carry["text"].split("\n\n", 1)[-1]
            section    = last["section"]
            header     = f"Section: {section}" if section else "Section: (document start)"
            result[-1] = {
                **last,
                "text":    f"{header}\n\n{last_body}\n\n{carry_body}",
                "page_end": carry["page_end"] or last["page_end"],
            }
        else:
            result.append(carry)  # only chunk — keep it even if small

    # Re-index
    for i, c in enumerate(result):
        c["chunk_index"] = i

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    pages:         list[tuple],
    source:        str,
    max_chars:     int = 1500,
    min_chars:     int = 300,
    overlap_chars: int = 200,
) -> list[dict]:
    """
    Convert cleaned (page, markdown) pairs into a list of chunk dicts.

    Args:
        pages:         Output of cleaning.clean_pages().
        source:        File name — stored in chunk metadata.
        max_chars:     Maximum chunk size in characters.
        min_chars:     Chunks smaller than this are merged with their neighbour.
        overlap_chars: Characters of overlap when a section spans multiple chunks.

    Returns:
        List of chunk dicts ready for embedding and storage.
    """
    blocks = _extract_blocks(pages)
    chunks = _pack_blocks(blocks, source, max_chars, overlap_chars)
    chunks = _merge_small(chunks, min_chars)
    return chunks
