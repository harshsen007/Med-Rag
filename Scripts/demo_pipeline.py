"""
Interactive walkthrough script to see exactly what each function in MedRAG takes and returns.
Run with:
    uv run python demo_pipeline.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.parsing import parse_file, count_images
from src.cleaning import clean_pages
from src.chunking import chunk_document

def main():
    pdf_path = _ROOT / "Input_docs/clinical_medicine_ashok_chandra.pdf"
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    print("=" * 70)
    print("STAGE 1: PARSING (src/parsing.py -> parse_file & count_images)")
    print("=" * 70)
    print(f"INPUT:  Path = {pdf_path}")
    
    img_info = count_images(pdf_path)
    print(f"IMAGES: Total embedded images found: {img_info['total_images']}")
    if img_info['page_images']:
        print(f"        Pages with images: {img_info['page_images']}")

    pages = parse_file(pdf_path, ocr_enabled=False)
    
    print(f"OUTPUT: list of tuples: [ (page_num, text), ... ]")
    print(f"        Total pages parsed: {len(pages)}")
    print(f"        Type: {type(pages)}, First item type: {type(pages[0])}")
    print(f"--- Sample from Page {pages[0][0]} (first 250 characters) ---")
    print(pages[0][1][:250].strip())
    print("-" * 70)

    print("\n" + "=" * 70)
    print("STAGE 2: CLEANING (src/cleaning.py -> clean_pages)")
    print("=" * 70)
    print(f"INPUT:  {len(pages)} raw page tuples")
    
    cleaned = clean_pages(pages)
    
    print(f"OUTPUT: {len(cleaned)} cleaned page tuples")
    print(f"        (Normalised whitespace, fixed broken ligatures, stripped running headers)")
    print(f"--- Sample from Cleaned Page {cleaned[0][0]} (first 250 characters) ---")
    print(cleaned[0][1][:250].strip())
    print("-" * 70)

    print("\n" + "=" * 70)
    print("STAGE 3: CHUNKING (src/chunking.py -> chunk_document)")
    print("=" * 70)
    print(f"INPUT:  {len(cleaned)} pages + source='{pdf_path.name}'")
    
    chunks = chunk_document(cleaned, source=pdf_path.name)
    
    print(f"OUTPUT: list of {len(chunks)} chunk dictionaries")
    if chunks:
        sample = chunks[0]
        print(f"\nExample Chunk Dict Structure (Chunk 0):")
        print(f"  source      : {sample['source']}")
        print(f"  doc_type    : {sample['doc_type']}")
        print(f"  section     : {sample['section']}")
        print(f"  pages       : {sample['page_start']} to {sample['page_end']}")
        print(f"  chunk_index : {sample['chunk_index']}")
        print(f"  char_length : {len(sample['text'])}")
        print(f"  text preview:\n{'-'*30}\n{sample['text'][:300]}...\n{'-'*30}")
    print("=" * 70)

if __name__ == "__main__":
    main()
