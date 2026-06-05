import pandas as pd
from semantic_scholar_fetcher import fetch_semantic_scholar_data
import time
import os


def re_enrich_master():
    """
    Takes the existing master_papers.csv and re-fetches Semantic Scholar data
    with the new expanded field set (h-index, references, venue).
    Strips arxiv version suffix (v1, v2) before querying Semantic Scholar
    but preserves original ID for merge.
    """
    df = pd.read_csv("../data/master_papers.csv")
    print(f"Loaded {len(df)} existing papers")
    
    # drop any existing SS columns (including duplicate _x/_y variants)
    cols_to_drop = [
        "citation_count", "influential_citation_count", "affiliations",
        "reference_count", "venue", "mean_h_index", "max_h_index",
        "reference_count_x", "venue_x", "mean_h_index_x", "max_h_index_x",
        "reference_count_y", "venue_y", "mean_h_index_y", "max_h_index_y"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    enriched_rows = []
    for i, row in df.iterrows():
        original_id = row["arxiv_id"]
        stripped_id = original_id.split("v")[0]
        print(f"  ({i+1}/{len(df)}) {stripped_id}")
        
        ss_data = fetch_semantic_scholar_data(stripped_id)
        
        if ss_data:
            ss_data["arxiv_id"] = original_id  # preserve original for merge
            enriched_rows.append(ss_data)
        else:
            enriched_rows.append({
                "arxiv_id": original_id,
                "citation_count": 0,
                "influential_citation_count": 0,
                "reference_count": 0,
                "venue": "",
                "mean_h_index": 0,
                "max_h_index": 0,
                "affiliations": ""
            })
        
        time.sleep(1)
    
    enriched = pd.DataFrame(enriched_rows)
    merged = df.merge(enriched, on="arxiv_id", how="left")
    
    # fill missing values
    for col in ["citation_count", "influential_citation_count", "reference_count"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in ["mean_h_index", "max_h_index"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    for col in ["venue", "affiliations"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")
    
    merged.to_csv("../data/master_papers.csv", index=False)
    print(f"\nSaved re-enriched master with columns: {merged.columns.tolist()}")
    print(f"Total papers: {len(merged)}")
    
    return merged


if __name__ == "__main__":
    re_enrich_master()