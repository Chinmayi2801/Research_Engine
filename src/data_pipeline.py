import pandas as pd
import re
import os
from datetime import datetime
from arxiv_fetcher import fetch_arxiv_papers
from semantic_scholar_fetcher import fetch_semantic_scholar_data
import time


def clean_abstract(text):
    """
    Remove LaTeX artifacts and normalize whitespace in abstract text.
    """
    if not isinstance(text, str):
        return ""
    
    # remove LaTeX inline math like $...$
    text = re.sub(r'\$[^$]*\$', '', text)
    
    # remove LaTeX commands like \textbf{...}, \cite{...}, \emph{...}
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    
    # remove standalone LaTeX commands like \alpha, \beta, \\
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # remove curly braces leftover
    text = re.sub(r'[{}]', '', text)
    
    # collapse multiple whitespaces into one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def standardize_date(date_str):
    """
    Convert date string to YYYY-MM-DD format.
    """
    if not isinstance(date_str, str):
        return None
    
    try:
        # arxiv returns dates as YYYY-MM-DD already, but we standardize just in case
        return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def clean_dataframe(df):
    """
    Apply all cleaning steps to the dataframe.
    """
    print(f"\nCleaning data...")
    print(f"Starting with {len(df)} rows")
    
    # drop rows with missing abstracts
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip() != ""]
    print(f"After removing empty abstracts: {len(df)} rows")
    
    # clean abstracts
    df["abstract"] = df["abstract"].apply(clean_abstract)
    df["title"] = df["title"].apply(clean_abstract)
    
    # standardize dates
    df["published_date"] = df["published_date"].apply(standardize_date)
    df = df.dropna(subset=["published_date"])
    print(f"After date validation: {len(df)} rows")
    
    # remove duplicates by arxiv_id
    df = df.drop_duplicates(subset=["arxiv_id"], keep="first")
    print(f"After deduplication: {len(df)} rows")
    
    # fill missing citation values with 0
    df["citation_count"] = df["citation_count"].fillna(0).astype(int)
    df["influential_citation_count"] = df["influential_citation_count"].fillna(0).astype(int)
    df["reference_count"] = df["reference_count"].fillna(0).astype(int)
    df["venue"] = df["venue"].fillna("")
    df["mean_h_index"] = df["mean_h_index"].fillna(0)
    df["max_h_index"] = df["max_h_index"].fillna(0)
    df["affiliations"] = df["affiliations"].fillna("")
    
    return df


def enrich_with_semantic_scholar(df):
    """
    Take arxiv dataframe and add Semantic Scholar columns.
    """
    print(f"\nEnriching {len(df)} papers with Semantic Scholar data...")
    
    enriched_rows = []
    
    for i, row in df.iterrows():
        arxiv_id = row["arxiv_id"]
        print(f"  ({i+1}/{len(df)}) {arxiv_id}")
        
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
        
        time.sleep(1)
    
    enriched_df = pd.DataFrame(enriched_rows)
    merged = df.merge(enriched_df, on="arxiv_id", how="left")
    return merged


def run_pipeline(topic, max_results=200, save_path=None):
    """
    End-to-end pipeline: fetch arXiv, enrich with Semantic Scholar, clean, save.
    """
    print(f"\n{'='*60}")
    print(f"Running pipeline for topic: {topic}")
    print(f"{'='*60}")
    
    # step 1: fetch from arxiv
    arxiv_df = fetch_arxiv_papers(topic, max_results=max_results)
    
    if arxiv_df is None or len(arxiv_df) == 0:
        print("No papers fetched. Aborting.")
        return None
    
    # step 2: enrich with semantic scholar
    enriched_df = enrich_with_semantic_scholar(arxiv_df)
    
    # step 3: clean
    cleaned_df = clean_dataframe(enriched_df)
    
    # step 4: save
    os.makedirs("../data", exist_ok=True)
    
    if save_path is None:
        clean_topic = topic.replace(" ", "_").lower()
        save_path = f"../data/{clean_topic}_master.csv"
    
    cleaned_df.to_csv(save_path, index=False)
    print(f"\nSaved {len(cleaned_df)} cleaned papers to {save_path}")
    
    return cleaned_df


def merge_all_topic_files(output_path="../data/master_papers.csv"):
    """
    Merges all topic-specific master CSVs in the data folder into one master file.
    """
    data_dir = "../data"
    all_files = [f for f in os.listdir(data_dir) if f.endswith("_master.csv")]
    
    if not all_files:
        print("No topic master files found to merge.")
        return None
    
    print(f"\nMerging {len(all_files)} topic files into one master CSV...")
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(os.path.join(data_dir, f))
        dfs.append(df)
    
    master = pd.concat(dfs, ignore_index=True)
    
    # deduplicate across topics
    master = master.drop_duplicates(subset=["arxiv_id"], keep="first")
    
    master.to_csv(output_path, index=False)
    print(f"Master file saved to {output_path}")
    print(f"Total unique papers: {len(master)}")
    
    return master


if __name__ == "__main__":
    # define your topics here
    topics = [
        "graph neural networks",   
    ]
    
    # fetch each topic
    for topic in topics:
        
        run_pipeline(topic, max_results=200)
        print("Sleeping 10 seconds before next topic...")
        time.sleep(10)
    
    
    
    # merge everything into master_papers.csv
    merge_all_topic_files()