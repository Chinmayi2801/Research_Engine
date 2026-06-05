import time
import pandas as pd
from arxiv_fetcher import fetch_arxiv_papers_by_date_range
from data_pipeline import enrich_with_semantic_scholar, clean_dataframe
import os
import sys


def fetch_one_topic(topic):
    """
    Fetches papers for a single topic between 2019-01-01 and 2020-12-31,
    enriches them with Semantic Scholar data, and saves as a topic-specific CSV.
    """
    start_date = "2019-01-01"
    end_date = "2020-12-31"
    
    print(f"\n=== Fetching {topic} ({start_date} to {end_date}) ===")
    
    df = fetch_arxiv_papers_by_date_range(topic, start_date, end_date, max_results=200)
    
    if df is None or len(df) == 0:
        print("No papers fetched. Aborting.")
        return None
    
    print(f"Got {len(df)} papers from arXiv. Enriching with Semantic Scholar...")
    
    enriched = enrich_with_semantic_scholar(df)
    cleaned = clean_dataframe(enriched)
    
    os.makedirs("../data", exist_ok=True)
    clean_topic = topic.replace(" ", "_").lower()
    save_path = f"../data/historical_{clean_topic}.csv"
    cleaned.to_csv(save_path, index=False)
    
    print(f"\nSaved {len(cleaned)} cleaned papers to {save_path}")
    return cleaned


def merge_historical():
    """
    Merges all historical_*.csv files in /data into one historical_master.csv.
    """
    data_dir = "../data"
    files = [f for f in os.listdir(data_dir) if f.startswith("historical_") and f.endswith(".csv") and f != "historical_master.csv"]
    
    if not files:
        print("No historical topic files found.")
        return None
    
    print(f"\nMerging {len(files)} historical topic files...")
    
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        dfs.append(df)
    
    master = pd.concat(dfs, ignore_index=True)
    master = master.drop_duplicates(subset=["arxiv_id"], keep="first")
    
    master.to_csv(os.path.join(data_dir, "historical_master.csv"), index=False)
    print(f"Saved historical_master.csv with {len(master)} unique papers")
    return master


if __name__ == "__main__":
    # called as: python fetch_historical.py <topic_name>
    # or: python fetch_historical.py merge
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_historical.py <topic>")
        print("       python fetch_historical.py merge")
        sys.exit(1)
    
    arg = " ".join(sys.argv[1:]).lower()
    
    if arg == "merge":
        merge_historical()
    else:
        fetch_one_topic(arg)