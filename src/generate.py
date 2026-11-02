"""
src/generate.py — Generate an answer from a question + retrieved chunks.

Public function:
  answer(question, chunks=None) → str

If chunks is None, retrieve() is called automatically.
The LLM sees: system prompt + user prompt with the numbered chunks + question.
"""

from openai import OpenAI
from src.config import settings
from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src import retrieve as _retrieve_module


def answer(question: str, chunks: list[dict] | None = None) -> str:
    """
    Generate a cited answer to the question.

    Args:
        question: The user's question.
        chunks:   Pre-retrieved chunk dicts (optional).
                  If None, retrieve() is called automatically.

    Returns:
        The LLM's answer as a string.
    """
    if chunks is None:
        chunks = _retrieve_module.retrieve(question)

    # Build the formatted chunk block for the prompt
    chunk_texts = []
    for i, chunk in enumerate(chunks, start=1):
        section  = chunk.get("section", "")
        doc_type = chunk.get("doc_type", "")

        if doc_type == "pubmed":
            # Richer citation header for PubMed chunks — the LLM copies this verbatim.
            authors   = chunk.get("authors", [])
            pub_year  = chunk.get("pub_year", "")
            journal   = chunk.get("journal", "")
            pmid      = chunk.get("pmid", "")
            doi       = chunk.get("doi", "")

            # First author + "et al." if more than one author
            if len(authors) > 1:
                authors_short = f"{authors[0]} et al."
            elif authors:
                authors_short = authors[0]
            else:
                authors_short = "Unknown"

            year_str = f" ({pub_year})" if pub_year else ""
            journal_str = f" — {journal}" if journal else ""
            pmid_str = f" | PMID: {pmid}" if pmid else ""
            doi_str  = f" | DOI: {doi}" if doi else ""

            header = (
                f"[{i}] Source: {authors_short}{year_str}{journal_str}"
                f"{pmid_str}{doi_str} | Section: {section}"
            )
        else:
            # Standard header for PDF / DOCX / Markdown sources
            source = chunk.get("source", "unknown")
            pg_s   = chunk.get("page_start")
            pg_e   = chunk.get("page_end")

            # Page label: "p.12" or "p.12-14" or "n/a"
            if pg_s and pg_e and pg_s != pg_e:
                page_label = f"p.{pg_s}-{pg_e}"
            elif pg_s:
                page_label = f"p.{pg_s}"
            else:
                page_label = "n/a"

            header = f"[{i}] Source: {source} | Section: {section} | Page: {page_label}"

        chunk_texts.append(f"{header}\n{chunk['text']}")

    formatted_chunks = "\n\n---\n\n".join(chunk_texts)

    # Fill the user prompt template
    user_message = USER_PROMPT_TEMPLATE.format(
        chunks=formatted_chunks,
        question=question,
    )

    # Call the NVIDIA-hosted LLM via the OpenAI-compatible API
    client = OpenAI(
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )

    return response.choices[0].message.content
