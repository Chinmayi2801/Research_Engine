import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv
import os

# load .env from project root (one level up from src)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

print(f"DEBUG: API_KEY loaded: {API_KEY[:10] if API_KEY else 'NONE'}...")

def fetch_semantic_scholar_data(arxiv_id):
    """
    Takes an arxiv ID and fetches citation and affiliation data
    from Semantic Scholar API.
    """

    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    
    params = {
        "fields": "citationCount,influentialCitationCount,authors.affiliations"
    }
    
    try:
        headers = {"x-api-key": API_KEY}
        response = requests.get(url, params=params, headers=headers)


        
        if response.status_code == 200:
            data = response.json()
            
            citation_count = data.get("citationCount", 0)
            influential_citation_count = data.get("influentialCitationCount", 0)
            
            # extract all unique affiliations across all authors
            affiliations = []
            for author in data.get("authors", []):
                for affiliation in author.get("affiliations", []):
                    if affiliation and affiliation not in affiliations:
                        affiliations.append(affiliation)
            
            affiliations_str = ", ".join(affiliations)
            
            return {
                "arxiv_id": arxiv_id,
                "citation_count": citation_count,
                "influential_citation_count": influential_citation_count,
                "affiliations": affiliations_str
            }
        
        elif response.status_code == 404:
            # paper not found on semantic scholar
            return {
                "arxiv_id": arxiv_id,
                "citation_count": 0,
                "influential_citation_count": 0,
                "affiliations": ""
            }
        
        elif response.status_code == 429:
            print(f"DEBUG: status={response.status_code}, response={response.text[:200]}")
            # rate limited, wait and retry
            print(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return fetch_semantic_scholar_data(arxiv_id)
        
        else:
            print(f"Unexpected status {response.status_code} for {arxiv_id}")
            return None
    
    except Exception as e:
        print(f"Error fetching {arxiv_id}: {e}")
        return None


def enrich_papers(arxiv_csv_path):
    """
    Takes the arxiv CSV, enriches each paper with Semantic Scholar data,
    and saves a merged master CSV.
    """
    
    print(f"Loading papers from {arxiv_csv_path}")
    df = pd.read_csv(arxiv_csv_path)
    print(f"Loaded {len(df)} papers")
    
    enriched_rows = []
    
    for i, row in df.iterrows():
        arxiv_id = row["arxiv_id"]
        print(f"Fetching Semantic Scholar data for {arxiv_id} ({i+1}/{len(df)})")
        
        ss_data = fetch_semantic_scholar_data(arxiv_id)
        
        if ss_data:
            enriched_rows.append(ss_data)
        else:
            enriched_rows.append({
                "arxiv_id": arxiv_id,
                "citation_count": 0,
                "influential_citation_count": 0,
                "affiliations": ""
            })
        
        # be polite to the API, wait 1 second between requests
        time.sleep(3)
    
    enriched_df = pd.DataFrame(enriched_rows)
    
    # merge with original arxiv data
    master_df = df.merge(enriched_df, on="arxiv_id", how="left")
    
    # save master CSV
    os.makedirs("../data", exist_ok=True)
    master_path = "../data/master_papers.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\nMaster CSV saved to {master_path}")
    print(f"Columns: {master_df.columns.tolist()}")
    print(master_df.head())
    
    return master_df


if __name__ == "__main__":
    arxiv_csv = "../data/machine_learning_papers.csv"
    master_df = enrich_papers(arxiv_csv)