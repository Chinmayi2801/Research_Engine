"""
Fetches historical papers (2019-2020) directly from Semantic Scholar's
/paper/search endpoint. Bypasses arXiv entirely.

Produces CSVs with the same column schema as master_papers.csv so that
train_pais_model.py and downstream scripts work without modification.

Usage:
    python fetch_historical_ss.py "machine learning"      # one topic
    python fetch_historical_ss.py all                     # ML, CV, NLP
    python fetch_historical_ss.py merge                   # merge into master
"""

import os
import sys
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# load .env from project root (one level up from src/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

print(f"DEBUG: API_KEY loaded: {API_KEY[:10] if API_KEY else 'NONE'}...")

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# fields we need — must match what train_pais_model.py expects
FIELDS = (
    "title,abstract,authors.name,authors.hIndex,authors.affiliations,"
    "year,publicationDate,citationCount,influentialCitationCount,"
    "referenceCount,venue,externalIds"
)


def fetch_papers_by_search(topic, year_range="2019-2020", target_count=200):
    """
    Paginated search call. Returns a list of raw paper dicts.
    """
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    all_papers = []
    offset = 0
    batch_size = 100  # max allowed by the endpoint

    print(f"\n=== Searching: '{topic}' year={year_range} target={target_count} ===")

    while len(all_papers) < target_count:
        params = {
            "query": topic,
            "year": year_range,
            "limit": min(batch_size, target_count - len(all_papers)),
            "offset": offset,
            "fields": FIELDS,
        }

        try:
            response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)

            if response.status_code == 429:
                print("Rate limited. Waiting 60s...")
                time.sleep(60)
                continue

            if response.status_code != 200:
                print(f"Status {response.status_code}: {response.text[:300]}")
                break

            data = response.json()
            results = data.get("data", [])

            if not results:
                print("No more results.")
                break

            all_papers.extend(results)
            print(f"Fetched {len(results)} papers (total so far: {len(all_papers)})")

            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = next_offset

            # polite delay between paginated calls
            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            break

    all_papers = all_papers[:target_count]
    print(f"Final raw count for '{topic}': {len(all_papers)}")
    return all_papers


def parse_paper(raw, topic):
    """
    Flatten a Semantic Scholar response into the schema used by master_papers.csv.
    h-index aggregation is done across ALL authors (the bug in the original
    enrichment script is fixed here).
    """
    if raw is None:
        return None

    authors_data = raw.get("authors") or []

    # author names
    author_names = [a.get("name", "") for a in authors_data if a.get("name")]
    authors_str = ", ".join(author_names)

    # h-indices across ALL authors
    h_indices = [a.get("hIndex") for a in authors_data if a.get("hIndex") is not None]
    mean_h = sum(h_indices) / len(h_indices) if h_indices else 0
    max_h = max(h_indices) if h_indices else 0

    # deduped affiliations
    affiliations = []
    for a in authors_data:
        for aff in (a.get("affiliations") or []):
            if aff and aff not in affiliations:
                affiliations.append(aff)
    affiliations_str = ", ".join(affiliations)

    # arxiv id if Semantic Scholar has it indexed
    external_ids = raw.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv", "") or ""

    # published_date — fall back to Jan 1 of the year if exact date missing
    pub_date = raw.get("publicationDate")
    if not pub_date:
        year = raw.get("year")
        pub_date = f"{year}-01-01" if year else ""

    return {
        "arxiv_id": arxiv_id,
        "paper_id": raw.get("paperId", ""),
        "title": raw.get("title", "") or "",
        "abstract": raw.get("abstract", "") or "",
        "authors": authors_str,
        "published_date": pub_date,
        "topic": topic,
        "citation_count": raw.get("citationCount", 0) or 0,
        "influential_citation_count": raw.get("influentialCitationCount", 0) or 0,
        "reference_count": raw.get("referenceCount", 0) or 0,
        "venue": raw.get("venue", "") or "",
        "mean_h_index": mean_h,
        "max_h_index": max_h,
        "affiliations": affiliations_str,
    }


def fetch_topic_historical(topic, year_range="2019-2020", target_count=200):
    """
    Fetch, parse, dedupe, save one topic's historical CSV.
    """
    raw_papers = fetch_papers_by_search(topic, year_range, target_count)

    rows = []
    for raw in raw_papers:
        parsed = parse_paper(raw, topic)
        # require title and abstract for downstream embedding
        if parsed and parsed["title"] and parsed["abstract"]:
            rows.append(parsed)

    df = pd.DataFrame(rows)
    if "paper_id" in df.columns:
        df = df.drop_duplicates(subset=["paper_id"], keep="first")

    os.makedirs("../data", exist_ok=True)
    clean_topic = topic.replace(" ", "_").lower()
    save_path = f"../data/historical_{clean_topic}.csv"
    df.to_csv(save_path, index=False)

    print(f"\nSaved {len(df)} papers to {save_path}")
    if len(df) > 0:
        nonzero = (df["citation_count"] > 0).sum()
        print(f"  Papers with non-zero citations: {nonzero}/{len(df)}")
        print(f"  Mean citation count: {df['citation_count'].mean():.1f}")
        print(f"  Median citation count: {df['citation_count'].median():.1f}")
        print(f"  Papers with h-index data: {(df['mean_h_index'] > 0).sum()}/{len(df)}")
    return df


def merge_historical():
    """
    Merges historical_*.csv into historical_master.csv.
    """
    data_dir = "../data"
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith("historical_") and f.endswith(".csv")
        and f != "historical_master.csv"
    ]

    if not files:
        print("No historical topic files found.")
        return None

    print(f"\nMerging {len(files)} files: {files}")
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
    master = pd.concat(dfs, ignore_index=True)
    if "paper_id" in master.columns:
        master = master.drop_duplicates(subset=["paper_id"], keep="first")

    save_path = os.path.join(data_dir, "historical_master.csv")
    master.to_csv(save_path, index=False)

    print(f"\nSaved historical_master.csv with {len(master)} unique papers")
    nonzero = (master["citation_count"] > 0).sum()
    print(f"  Papers with non-zero citations: {nonzero}/{len(master)}")
    print(f"  Mean citation count: {master['citation_count'].mean():.1f}")
    print(f"  Median citation count: {master['citation_count'].median():.1f}")
    print(f"  Papers with h-index data: {(master['mean_h_index'] > 0).sum()}/{len(master)}")

    return master


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python fetch_historical_ss.py "machine learning"   # one topic')
        print("  python fetch_historical_ss.py all                  # ML, CV, NLP")
        print("  python fetch_historical_ss.py merge                # merge all")
        sys.exit(1)

    arg = " ".join(sys.argv[1:]).lower()

    if arg == "merge":
        merge_historical()
    elif arg == "all":
        topics = ["machine learning", "computer vision", "natural language processing"]
        for t in topics:
            fetch_topic_historical(t)
            time.sleep(5)  # buffer between topics
        merge_historical()
    else:
        fetch_topic_historical(arg)