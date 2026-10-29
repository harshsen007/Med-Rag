"""
src/cleaning.py — Fix common PDF text artefacts before chunking.

Steps (in order):
  1. Replace \\x00 (broken "fi" ligature in some PDFs) with "fi".
  2. Replace private-use Unicode chars (\\uf000–\\uf8ff) with "-".
  3. Remove lines that are just a bare page number like "42" or "- 42 -".
  4. Remove lines that repeat suspiciously often (headers / footers).
  5. Collapse runs of blank lines and strip trailing whitespace.
"""

import re
from collections import Counter


# Matches a line that is only a page number  e.g. "42" or "— 42 —"
_BARE_PAGE_NUM = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


def clean_text(text: str) -> str:
    """Apply all cleaning steps to one Markdown string."""

    # Step 1 & 2: character fixes
    text = text.replace("\x00", "fi")
    text = re.sub(r"[\uf000-\uf8ff]", "-", text)

    lines = text.split("\n")

    # Step 3: remove bare page numbers
    lines = [ln for ln in lines if not _BARE_PAGE_NUM.match(ln)]

    # Step 4: remove repeated header/footer lines
    lines = _remove_repeated_lines(lines)

    # Step 5: normalise whitespace
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                   # multiple blank lines
    return text.strip()


def _remove_repeated_lines(lines: list[str], min_repeats: int = 4) -> list[str]:
    """
    Drop lines that appear >= min_repeats times — these are usually
    headers or footers repeated on every page.
    Only checks lines with at least 5 characters (to avoid removing bullets, etc.).
    """
    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if len(s) >= 5)
    repeated = {s for s, n in counts.items() if n >= min_repeats}
    return [ln for ln, s in zip(lines, stripped) if s not in repeated]


def clean_pages(pages: list[tuple]) -> list[tuple]:
    """
    Clean every page returned by the parser.
    Drops pages that become empty after cleaning.

    Input / output:  list of (page_number_or_None, markdown_text)
    """
    cleaned = []
    for page_num, text in pages:
        result = clean_text(text)
        if result:  # skip pages that are now empty
            cleaned.append((page_num, result))
    return cleaned
