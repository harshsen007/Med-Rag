"""
src/parsing.py — Turn any supported file into a list of (page, markdown) pairs.

Supported: .pdf  .md  .docx  .html  .htm

Returns:  list[ tuple[int | None, str] ]
  - page is 1-based for PDFs, None for everything else.
  - str is the page content as Markdown.
"""

import shutil
import warnings
from pathlib import Path


def parse_file(path: Path, ocr_enabled: bool = True, ocr_language: str = "eng"):
    """
    Main entry point.  Detect format by extension, dispatch to the right parser.
    Returns a list of (page_number_or_None, markdown_text).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(path, ocr_enabled, ocr_language)
    elif ext in {".md", ".markdown"}:
        return _parse_markdown(path)
    elif ext in {".docx", ".html", ".htm"}:
        return _parse_with_markitdown(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use: .pdf .md .docx .html")


# ── PDF ───────────────────────────────────────────────────────────────────────

def _parse_pdf(path: Path, ocr_enabled: bool, ocr_language: str):
    """One Markdown string per page. OCR only if the page has no text layer."""
    import pymupdf4llm
    import pymupdf

    doc = pymupdf.open(str(path))
    total_pages = len(doc)
    total_images = sum(len(page.get_images()) for page in doc)
    doc.close()

    if total_images > 0:
        print(f"   [PDF Parser] Detected {total_images} embedded image(s) across {total_pages} page(s).")
    else:
        print(f"   [PDF Parser] No embedded images found across {total_pages} page(s).")

    results = []
    tesseract_missing_warned = False

    for page_index in range(total_pages):
        page_num = page_index + 1  # 1-based

        # pymupdf4llm uses 0-based page indices
        md_text = pymupdf4llm.to_markdown(str(path), pages=[page_index], show_progress=False)

        # If the page has no text layer and OCR is on, try Tesseract
        if ocr_enabled and _is_empty(md_text):
            if not shutil.which("tesseract"):
                if not tesseract_missing_warned:
                    warnings.warn(
                        "OCR_ENABLED=True but tesseract is not installed — skipping OCR.",
                        RuntimeWarning,
                    )
                    tesseract_missing_warned = True
            else:
                md_text = _ocr_page(path, page_index, ocr_language)

        results.append((page_num, md_text))

    return results


def _is_empty(text: str) -> bool:
    """True if the page has fewer than 50 non-space characters (probably image-only)."""
    return len(text.replace(" ", "").replace("\n", "")) < 50


def _ocr_page(pdf_path: Path, page_index: int, language: str) -> str:
    """Render one PDF page to PNG and run Tesseract on it."""
    import pymupdf
    import tempfile, subprocess, os

    doc = pymupdf.open(str(pdf_path))
    page = doc[page_index]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), alpha=False)

    with tempfile.TemporaryDirectory() as tmp:
        img = os.path.join(tmp, "page.png")
        out = os.path.join(tmp, "out")
        pix.save(img)
        subprocess.run(["tesseract", img, out, "-l", language], check=True, capture_output=True)
        return open(out + ".txt").read()


# ── DOCX / HTML ───────────────────────────────────────────────────────────────

def _parse_with_markitdown(path: Path):
    """markitdown handles DOCX and HTML → Markdown.  No page numbers."""
    from markitdown import MarkItDown
    result = MarkItDown().convert(str(path))
    return [(None, result.text_content)]


# ── Markdown ──────────────────────────────────────────────────────────────────

def _parse_markdown(path: Path):
    """Read a .md file as-is.  No page numbers."""
    return [(None, path.read_text(encoding="utf-8"))]


# ── Image Inspection ─────────────────────────────────────────────────────────

def count_images(path: Path) -> dict:
    """
    Returns image count details for a PDF:
    {
      "total_images": int,
      "page_images": {page_num: count}  # only pages with images > 0
    }
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return {"total_images": 0, "page_images": {}}

    import pymupdf

    doc = pymupdf.open(str(path))
    page_images = {}
    total = 0
    for i, page in enumerate(doc):
        imgs = page.get_images()
        if imgs:
            page_images[i + 1] = len(imgs)
            total += len(imgs)
    doc.close()
    return {"total_images": total, "page_images": page_images}

