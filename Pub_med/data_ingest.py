"""
Pub_med/data_ingest.py — Debug/exploration script for the PubMed Entrez API.

Prints the raw esearch JSON and the parsed BioPython dict for each article.
NOT connected to the embedding pipeline — use pubmed_ingest.py for that.

Changes from original:
  - term: "all[sb] AND hasabstract"  (hasabstract = no field tag, correct PubMed filter)
  - datetype: "edat"                 (Entrez date — reliable for "recently added")
  - retmax: 100                      (explicit cap; default 20 is a silent ceiling)
"""

import json
from Bio import Entrez
from src.config import settings

# Configure credentials
Entrez.email = settings.PUB_MED_ACC
Entrez.api_key = settings.PUB_MED_API_KEY

print("--- 1. FETCHING RAW ESEARCH (JSON) RESPONSE ---")
search_handle = Entrez.esearch(
    db="pubmed",
    term="all[sb] AND hasabstract",   # hasabstract: no [text] tag — correct PubMed filter
    reldate=1,
    datetype="edat",                   # Entrez date: reliable for "recently added"
    retmax=100,                        # explicit cap; default 20 is a silent ceiling
    retmode="json",
)
raw_search_data = search_handle.read()
search_handle.close()

parsed_search_json = json.loads(raw_search_data)
print(json.dumps(parsed_search_json, indent=2))

pmid_list = parsed_search_json["esearchresult"]["idlist"]
print(f"\nFound {len(pmid_list)} new articles.")

if pmid_list:
    try:
        print("\n--- 2. FETCHING RAW EFETCH (XML) RESPONSE FOR ALL ARTICLES ---")
        fetch_handle = Entrez.efetch(db="pubmed", id=pmid_list, retmode="xml")
        raw_xml_data = fetch_handle.read()
        fetch_handle.close()

        print(raw_xml_data.decode("utf-8") if isinstance(raw_xml_data, bytes) else raw_xml_data)

        from io import BytesIO
        stream = BytesIO(raw_xml_data if isinstance(raw_xml_data, bytes) else raw_xml_data.encode("utf-8"))
        articles = Entrez.read(stream)

        print("\n--- 3. ITERATING THROUGH PARSED DICTIONARY ENTRIES ---")
        for i, article in enumerate(articles["PubmedArticle"]):
            print(f"\n=== RAW DICTIONARY FOR SINGLE ARTICLE {i + 1} ===")
            print(json.dumps(article, indent=2, default=str))

    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("No articles found to fetch.")
