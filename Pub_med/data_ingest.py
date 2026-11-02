# import json
# from Bio import Entrez
# from src.config import settings


# # Always configure your email and API key
# Entrez.email = settings.PUB_MED_ACC
# Entrez.api_key = settings.PUB_MED_API_KEY

# # 1. Search for the latest articles from the past 1 day
# search_handle = Entrez.esearch(db="pubmed", term="all[sb]", reldate=1, datetype="pdat")
# search_results = Entrez.read(search_handle)
# search_handle.close()

# pmid_list = search_results["IdList"]
# print(f"Found {len(pmid_list)} new articles.")

# # 2. Fetch details for those articles if any were found
# if pmid_list:
#     try:
#         # We pass the list of IDs directly. BioPython handles the comma separation automatically.
#         fetch_handle = Entrez.efetch(db="pubmed", id=pmid_list, retmode="xml")
#         articles = Entrez.read(fetch_handle)
#         fetch_handle.close()
        
#         # 3. Iterate through the dictionary to see the articles
#         for i, article in enumerate(articles['PubmedArticle']):
#             medline_citation = article['MedlineCitation']
#             article_meta = medline_citation['Article']
            
#             # Extract basic fields
#             pmid = medline_citation['PMID']
#             title = article_meta['ArticleTitle']
            
#             # Extract abstract safely (some articles don't have one yet)
#             abstract_text = "No abstract available."
#             if 'Abstract' in article_meta:
#                 abstract_text = " ".join(article_meta['Abstract']['AbstractText'])
            
#             print(f"\n--- Article {i+1} (PMID: {pmid}) ---")
#             print(f"Title: {title}")
#             print(f"Abstract: {abstract_text[:300]}...") # Truncating to 300 characters for readability
            
#     except Exception as e:
#         print(f"An error occurred while fetching details: {e}")
# else:
#     print("No articles found to fetch.")



import json  # Added to print clean, formatted JSON
from Bio import Entrez
from src.config import settings

# Configure credentials
Entrez.email = settings.PUB_MED_ACC
Entrez.api_key = settings.PUB_MED_API_KEY

print("--- 1. FETCHING RAW ESEARCH (JSON) RESPONSE ---")
# Force retmode="json" to get real JSON from the search API
search_handle = Entrez.esearch(db="pubmed", term="all[sb]", reldate=1, datetype="pdat", retmode="json")
raw_search_data = search_handle.read() # Read raw string before parsing
search_handle.close()

# Print the raw search JSON neatly
parsed_search_json = json.loads(raw_search_data)
print(json.dumps(parsed_search_json, indent=2))

# Extract IDs manually from the raw JSON structure
pmid_list = parsed_search_json["esearchresult"]["idlist"]
print(f"\nFound {len(pmid_list)} new articles.")

# 2. Fetch raw details for those articles if any were found
if pmid_list:
    try:
        print("\n--- 2. FETCHING RAW EFETCH (XML) RESPONSE FOR ALL ARTICLES ---")
        fetch_handle = Entrez.efetch(db="pubmed", id=pmid_list, retmode="xml")
        raw_xml_data = fetch_handle.read() # Read the complete raw XML string
        fetch_handle.close()
        
        # PRINT STATEMENT 1: See the entire raw response from the server
        print(raw_xml_data.decode('utf-8') if isinstance(raw_xml_data, bytes) else raw_xml_data)
        
        # Now convert to BioPython object to look at individual items
        # We use Entrez.read() on a new handle or parse the string using BioPython's internal parser
        from io import BytesIO
        stream = BytesIO(raw_xml_data if isinstance(raw_xml_data, bytes) else raw_xml_data.encode('utf-8'))
        articles = Entrez.read(stream)
        
        print("\n--- 3. ITERATING THROUGH PARSED DICTIONARY ENTRIES ---")
        for i, article in enumerate(articles['PubmedArticle']):
            print(f"\n=== RAW DICTIONARY FOR SINGLE ARTICLE {i+1} ===")
            # PRINT STATEMENT 2: This prints the Python dictionary version of a single article
            print(json.dumps(article, indent=2, default=str))
            
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("No articles found to fetch.")
