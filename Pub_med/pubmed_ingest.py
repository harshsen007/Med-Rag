"""
Pub_med/pubmed_ingest.py — Bridge between the PubMed Entrez API and the embedding pipeline.

Spike scope: fetches the first article with a valid abstract, builds one chunk dict,
upserts it into Qdrant, and runs a demo retrieve + generate loop to confirm end-to-end.

Public functions:
    fetch_articles()        → list of raw BioPython article dicts
    build_chunk(article)    → single chunk dict ready for upsert_chunks()
    ingest_first_article()  → fetch, build, upsert, print confirmation

Helper functions (pure, independently testable):
    _extract_doi(article)     → str   (two-source fallback)
    _extract_year(article)    → int   (three-source fallback)
    _extract_authors(article) → list[str]
    _build_abstract_body(abstract_list) → str  (handles labelled sections)
"""

import re
import sys
from io import BytesIO

from Bio import Entrez

from src.config import settings
from src import vector_store
import src.generate as _generate_module
import importlib
_retrieve_module = importlib.import_module("src.retrieve")

# Configure Entrez credentials from settings
Entrez.email = settings.PUB_MED_ACC
Entrez.api_key = settings.PUB_MED_API_KEY


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_articles(reldate: int = 1, retmax: int = 100) -> list[dict]:
    """
    Search PubMed for articles added in the last `reldate` days and return the
    parsed BioPython article dicts.

    Uses:
      term="all[sb] AND hasabstract"  — filters to records that have abstracts
      datetype="edat"                 — Entrez date (reliable for "recently added")
    """
    search_handle = Entrez.esearch(
        db="pubmed",
        term="all[sb] AND hasabstract",
        reldate=reldate,
        datetype="edat",
        retmax=retmax,
        retmode="json",
    )
    import json
    raw = search_handle.read()
    search_handle.close()
    pmid_list = json.loads(raw)["esearchresult"]["idlist"]

    if not pmid_list:
        print("No articles found for the given date range.")
        return []

    print(f"Found {len(pmid_list)} PMID(s). Fetching full records ...")
    fetch_handle = Entrez.efetch(db="pubmed", id=pmid_list, retmode="xml")
    raw_xml = fetch_handle.read()
    fetch_handle.close()

    stream = BytesIO(raw_xml if isinstance(raw_xml, bytes) else raw_xml.encode("utf-8"))
    parsed = Entrez.read(stream)
    return list(parsed.get("PubmedArticle", []))


# ── Helper extractors ─────────────────────────────────────────────────────────

def _extract_doi(article_meta: dict, pubmed_data: dict) -> str:
    """
    Extract the DOI string (without https://doi.org/ prefix).

    Accepts:
        article_meta — article["MedlineCitation"]["Article"]
        pubmed_data  — article["PubmedData"]

    Source 1: article_meta["ELocationID"] — filter by EIdType == "doi".
    Source 2: pubmed_data["ArticleIdList"] — filter by IdType == "doi".
    Returns "" if no DOI is found.
    """
    # Source 1: ELocationID on the Article
    for loc in article_meta.get("ELocationID", []):
        if getattr(loc, "attributes", {}).get("EIdType", "").lower() == "doi":
            return str(loc)
    # Source 2: ArticleIdList in PubmedData
    for aid in pubmed_data.get("ArticleIdList", []):
        if getattr(aid, "attributes", {}).get("IdType", "").lower() == "doi":
            return str(aid)
    return ""


def _extract_year(article_meta: dict) -> int:
    """
    Extract the publication year as an integer.

    Accepts article_meta — article["MedlineCitation"]["Article"].

    Fallback chain:
      1. Journal.JournalIssue.PubDate.Year
      2. Journal.JournalIssue.PubDate.MedlineDate  (e.g. "2026 Nov-Dec" → 2026)
      3. ArticleDate[0].Year  (electronic publication date)
    Returns 0 if no year can be determined.

    Stored as int so Qdrant range filters (gte/lte) work without casting.
    """
    try:
        pub_date = article_meta["Journal"]["JournalIssue"]["PubDate"]
        if "Year" in pub_date:
            return int(str(pub_date["Year"]))
        if "MedlineDate" in pub_date:
            m = re.search(r"\b(\d{4})\b", str(pub_date["MedlineDate"]))
            if m:
                return int(m.group(1))
    except (KeyError, TypeError, ValueError):
        pass

    # Fallback: electronic publication date
    try:
        article_dates = article_meta.get("ArticleDate", [])
        if article_dates and "Year" in article_dates[0]:
            return int(str(article_dates[0]["Year"]))
    except (KeyError, TypeError, ValueError):
        pass

    return 0


def _extract_authors(article_meta: dict) -> list[str]:
    """
    Return a list of formatted author name strings.

    Accepts the Article sub-dict (i.e. article["MedlineCitation"]["Article"]).

    Format: "LastName Initials"  (e.g. "Smith JA")
    Falls back to CollectiveName for corporate/group authors.
    Entries with neither LastName nor CollectiveName are skipped.
    """
    names = []
    for a in article_meta.get("AuthorList", []):
        last = str(a.get("LastName", "")).strip()
        initials = str(a.get("Initials", "")).strip()
        collective = str(a.get("CollectiveName", "")).strip()
        if last:
            names.append(f"{last} {initials}".strip())
        elif collective:
            names.append(collective)
    return names


def _build_abstract_body(abstract_list) -> str:
    """
    Join the AbstractText list into a single string.

    Structured abstracts have Label attributes (BACKGROUND, METHODS, etc.).
    Each labelled part is prefixed with its label.
    Unlabelled or UNLABELLED parts are included as plain text.
    """
    parts = []
    for part in abstract_list:
        label = getattr(part, "attributes", {}).get("Label", "")
        text = str(part).strip()
        if label and label.upper() not in ("", "UNLABELLED"):
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


# ── Chunk builder ─────────────────────────────────────────────────────────────

def build_chunk(article: dict) -> dict | None:
    """
    Convert a single BioPython article dict into one chunk dict.

    Returns None if the article has no Abstract (should be caught by the
    hasabstract filter, but this is a belt-and-suspenders guard).

    The chunk text is:
        Section: Abstract

        Title: <article title>

        <abstract body (with labels if structured)>

    All metadata fields are stored on this chunk so any retrieval hit can
    produce a complete citation without needing to look up sibling chunks.
    """
    medline = article["MedlineCitation"]
    article_meta = medline["Article"]
    pmid = str(medline["PMID"])

    # Post-fetch guard — hasabstract filter should prevent this, but be safe
    if "Abstract" not in article_meta:
        print(f"  Skipping PMID {pmid} — no Abstract key (post-fetch guard triggered)")
        return None

    abstract_list = article_meta["Abstract"].get("AbstractText", [])
    if not abstract_list:
        print(f"  Skipping PMID {pmid} — AbstractText is empty")
        return None

    title = str(article_meta.get("ArticleTitle", "")).strip()
    abstract_body = _build_abstract_body(abstract_list)

    # Compose the chunk text: title prepended to abstract body
    text = f"Section: Abstract\n\nTitle: {title}\n\n{abstract_body}"

    # Extract metadata — all helpers receive article_meta (already the Article sub-dict)
    authors  = _extract_authors(article_meta)
    doi      = _extract_doi(article_meta, article.get("PubmedData", {}))
    pub_year = _extract_year(article_meta)

    # Journal: prefer ISOAbbreviation, fall back to full Title
    journal_info = article_meta.get("Journal", {})
    journal = str(
        journal_info.get("ISOAbbreviation")
        or journal_info.get("Title", "")
    ).strip()

    # First author's first affiliation (or "" if absent)
    affiliation = ""
    author_list = article_meta.get("AuthorList", [])
    if author_list:
        aff_info = author_list[0].get("AffiliationInfo", [])
        if aff_info:
            affiliation = str(aff_info[0].get("Affiliation", "")).strip()

    return {
        # ── Standard pipeline fields ──────────────────────────────────────────
        "text":        text,
        "source":      f"pubmed::{pmid}",
        "doc_type":    "pubmed",
        "section":     "Abstract",
        "page_start":  None,
        "page_end":    None,
        "chunk_index": 0,
        # ── PubMed-specific fields ─────────────────────────────────────────────
        "pmid":        pmid,
        "title":       title,
        "authors":     authors,
        "affiliation": affiliation,
        "journal":     journal,
        "pub_year":    pub_year,   # int — supports Qdrant range filters
        "doi":         doi,
    }


# ── Main ingest function ──────────────────────────────────────────────────────

def ingest_first_article() -> dict | None:
    """
    Fetch articles, find the first one with a valid abstract, build its chunk,
    upsert into Qdrant, and return the chunk dict.

    Tries articles in order until one succeeds (in case early results have no
    abstract despite the hasabstract filter).
    """
    articles = fetch_articles()
    if not articles:
        print("No articles returned from PubMed.")
        return None

    chunk = None
    for i, article in enumerate(articles):
        print(f"Processing article {i + 1}/{len(articles)} ...")
        chunk = build_chunk(article)
        if chunk is not None:
            break

    if chunk is None:
        print("No articles with a valid abstract were found.")
        return None

    # Ensure the Qdrant collection exists
    vector_store.ensure_collection()

    # Skip if this PMID is already in Qdrant — no point re-embedding or re-writing
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    source_id = f"pubmed::{chunk['pmid']}"
    existing = vector_store._client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    if existing[0]:
        print(f"\n↺ Already in Qdrant — skipped (source={source_id})")
        return chunk

    # New article — embed and insert
    vector_store.upsert_chunks([chunk])

    pmid     = chunk["pmid"]
    title    = chunk["title"]
    authors  = chunk["authors"]
    journal  = chunk["journal"]
    pub_year = chunk["pub_year"]
    doi      = chunk["doi"]

    authors_display = authors[0] + " et al." if len(authors) > 1 else (authors[0] if authors else "Unknown")

    print(f"\n✓ Inserted")
    print(f"  PMID      : {pmid}")
    print(f"  Title     : {title[:80]}{'...' if len(title) > 80 else ''}")
    print(f"  Authors   : {authors_display}")
    print(f"  Journal   : {journal} ({pub_year})")
    print(f"  DOI       : {doi or '(none)'}")
    print(f"  Chunks    : 1  (chunk_index=0, source={source_id})")

    return chunk


# ── Demo block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 — Ingest first PubMed article")
    print("=" * 60)

    chunk = ingest_first_article()
    if chunk is None:
        sys.exit(1)

    question = f"What are the key findings of the study titled: {chunk['title'][:60]}?"

    print("\n" + "=" * 60)
    print("STEP 2 — Retrieve + Generate (end-to-end demo)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    retrieved = _retrieve_module.retrieve(question)
    print(f"Retrieved {len(retrieved)} chunk(s).")

    answer = _generate_module.answer(question, retrieved)
    print("\n--- Answer ---")
    print(answer)

    print("\n" + "=" * 60)
    print("STEP 3 — Rollback test (delete then re-ingest)")
    print("=" * 60)
    source = f"pubmed::{chunk['pmid']}"
    vector_store.delete_by_source(source)
    print(f"Deleted all points with source={source!r}")

    # Re-ingest — same PMID → same deterministic point ID → idempotent upsert
    print("\nRe-ingesting ...")
    chunk2 = ingest_first_article()
    if chunk2 and chunk2["pmid"] == chunk["pmid"]:
        print(f"\n✓ Idempotency confirmed — same PMID ({chunk['pmid']}) re-upserted cleanly.")
    else:
        print("\n⚠ Re-ingest returned a different article (API may have returned different results).")
