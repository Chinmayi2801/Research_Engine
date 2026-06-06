"""
Day 14: Engine — the orchestration layer for the research discovery platform.

Wraps FAISS search, PAIS scoring, topic modeling, RAG summarization, and
institution filtering into a clean interface for the Streamlit UI.

The UI should depend only on this module, not on the underlying components.
"""

import os
import pandas as pd
from rapidfuzz import fuzz

from search_engine import search_papers, find_similar_papers
from rag_summarizer import summarize_topic
from topic_trends import (
    get_topic_overview,
    get_topic_papers,
    get_topic_similarities,
)


PAPERS_WITH_TOPICS = "../models/papers_with_topics.csv"
PAPERS_WITH_PAIS = "../models/papers_with_pais.csv"
MASTER_PAPERS = "../data/master_papers.csv"


# ----------------------------------------------------------------------------
# DATA LOADING — single source of truth
# ----------------------------------------------------------------------------

def _load_combined_data():
    """
    Joins:
      - papers_with_topics.csv (titles, abstracts, authors, topic assignments)
      - papers_with_pais.csv (PAIS scores)
      - master_papers.csv (latest affiliations after re-enrichment)
    Returns one DataFrame keyed on arxiv_id.
    """
    topics_df = pd.read_csv(PAPERS_WITH_TOPICS)

    # join PAIS scores
    if os.path.exists(PAPERS_WITH_PAIS):
        pais_df = pd.read_csv(PAPERS_WITH_PAIS)
        pais_cols = ["arxiv_id", "pais_score", "pais_predicted_citations",
                     "reference_count", "mean_h_index", "max_h_index"]
        pais_subset = pais_df[[c for c in pais_cols if c in pais_df.columns]]
        topics_df = topics_df.merge(pais_subset, on="arxiv_id", how="left")

    # join fresh affiliations from master_papers.csv (post re-enrichment)
    if os.path.exists(MASTER_PAPERS):
        master_df = pd.read_csv(MASTER_PAPERS)
        if "affiliations" in master_df.columns:
            if "affiliations" in topics_df.columns:
                topics_df = topics_df.drop(columns=["affiliations"])
            topics_df = topics_df.merge(
                master_df[["arxiv_id", "affiliations"]],
                on="arxiv_id",
                how="left",
            )

    return topics_df


# ----------------------------------------------------------------------------
# 1. SEARCH WITH HYBRID RANKING (SIMILARITY + PAIS)
# ----------------------------------------------------------------------------

def search(query, top_k=10, similarity_weight=0.6, pais_weight=0.4):
    """
    Semantic search re-ranked by a hybrid of FAISS similarity and PAIS.

    The retrieval pulls 3x candidates so the re-ranking has room to work.
    Final score = similarity_weight * similarity + pais_weight * pais_score.
    Papers with no PAIS (insufficient feature data) get pais_score = 0 in the
    hybrid calculation, falling back to pure similarity ranking.
    """
    candidates = search_papers(query, top_k=top_k * 3)

    df = _load_combined_data()
    pais_subset = df[["arxiv_id", "pais_score", "topic"]]
    candidates = candidates.merge(pais_subset, on="arxiv_id", how="left")

    sim = candidates["similarity"].clip(0, 1)
    pais = candidates["pais_score"].fillna(0)
    candidates["hybrid_score"] = similarity_weight * sim + pais_weight * pais

    return candidates.sort_values("hybrid_score", ascending=False) \
                     .head(top_k).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. RELATED PAPERS
# ----------------------------------------------------------------------------

def get_related_papers(arxiv_id, top_k=10):
    """
    For a given paper, return the top-K most similar papers by embedding cosine
    similarity, joined with PAIS scores.
    """
    # use the original CSV ordering to map arxiv_id -> paper_idx
    topics_df = pd.read_csv(PAPERS_WITH_TOPICS)
    match = topics_df.index[topics_df["arxiv_id"] == arxiv_id]
    if len(match) == 0:
        raise ValueError(f"Paper {arxiv_id} not found in dataset")
    paper_idx = int(match[0])

    similar = find_similar_papers(paper_idx, top_k=top_k)

    if os.path.exists(PAPERS_WITH_PAIS):
        pais_df = pd.read_csv(PAPERS_WITH_PAIS)
        similar = similar.merge(
            pais_df[["arxiv_id", "pais_score"]],
            on="arxiv_id",
            how="left",
        )

    return similar


# ----------------------------------------------------------------------------
# 3. PAPER DETAILS
# ----------------------------------------------------------------------------

def get_paper_details(arxiv_id):
    """Return all available metadata for a single paper, or None if not found."""
    df = _load_combined_data()
    paper = df[df["arxiv_id"] == arxiv_id]
    if len(paper) == 0:
        return None
    return paper.iloc[0].to_dict()


# ----------------------------------------------------------------------------
# 4. RAG TOPIC SUMMARY (passthrough)
# ----------------------------------------------------------------------------

def get_topic_summary(query, top_k=8):
    """Delegate to rag_summarizer.summarize_topic (with caching)."""
    return summarize_topic(query, top_k=top_k)


# ----------------------------------------------------------------------------
# 5. INSTITUTION FILTER (fuzzy matching)
# ----------------------------------------------------------------------------

def filter_by_institution(institution_query, top_k=20, score_threshold=80):
    """
    Find papers where at least one author affiliation fuzzy-matches the query.
    Uses rapidfuzz.token_set_ratio, which handles abbreviations and word order.

    Args:
        institution_query: e.g. "MIT", "Stanford University", "iit bombay"
        top_k: max papers to return
        score_threshold: minimum fuzzy match score (0-100). 80 is moderately strict.

    Returns:
        DataFrame sorted by best affiliation match score, then PAIS descending.
    """
    df = _load_combined_data()
    query_lower = institution_query.lower().strip()

    def best_affiliation_score(affiliations):
        if not isinstance(affiliations, str) or not affiliations.strip():
            return 0
        affs = [a.strip() for a in affiliations.split(",")]
        scores = [fuzz.token_set_ratio(query_lower, a.lower()) for a in affs if a]
        return max(scores) if scores else 0

    df["affiliation_score"] = df["affiliations"].apply(best_affiliation_score)

    matches = df[df["affiliation_score"] >= score_threshold].copy()

    sort_cols = ["affiliation_score"]
    sort_ascending = [False]
    if "pais_score" in matches.columns:
        sort_cols.append("pais_score")
        sort_ascending.append(False)

    matches = matches.sort_values(
        by=sort_cols, ascending=sort_ascending, na_position="last"
    )

    cols = ["arxiv_id", "title", "authors", "affiliations",
            "published_date", "topic", "affiliation_score"]
    if "pais_score" in matches.columns:
        cols.append("pais_score")

    return matches[cols].head(top_k).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 6. TOPIC EXPLORATION (passthroughs to topic_trends)
# ----------------------------------------------------------------------------
# get_topic_overview, get_topic_papers, get_topic_similarities are imported
# at the top of this file and re-exposed here for the UI's convenience.


# ----------------------------------------------------------------------------
# TEST — exercises every public function
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: search('graph neural networks for drug discovery')")
    print("=" * 60)
    results = search("graph neural networks for drug discovery", top_k=5)
    for _, row in results.iterrows():
        sim = row["similarity"]
        pais = row.get("pais_score")
        pais_str = f"{pais:.3f}" if pd.notna(pais) else "N/A"
        print(f"  {row['title']}")
        print(f"    hybrid={row['hybrid_score']:.3f} | sim={sim:.3f} | pais={pais_str}")

    if len(results) > 0:
        first_id = results.iloc[0]["arxiv_id"]

        print("\n" + "=" * 60)
        print(f"TEST 2: get_related_papers({first_id})")
        print("=" * 60)
        related = get_related_papers(first_id, top_k=5)
        for _, row in related.iterrows():
            print(f"  {row['title']} (sim: {row['similarity']:.3f})")

        print("\n" + "=" * 60)
        print(f"TEST 3: get_paper_details({first_id})")
        print("=" * 60)
        details = get_paper_details(first_id)
        print(f"  Title: {details['title']}")
        print(f"  Authors: {details['authors']}")
        print(f"  Topic: {details.get('topic', 'N/A')}")
        print(f"  PAIS: {details.get('pais_score', 'N/A')}")
        affs = details.get("affiliations", "")
        affs_str = affs[:120] + "..." if isinstance(affs, str) and len(affs) > 120 else affs
        print(f"  Affiliations: {affs_str}")

    print("\n" + "=" * 60)
    print("TEST 4: filter_by_institution('MIT')")
    print("=" * 60)
    mit_papers = filter_by_institution("MIT", top_k=5, score_threshold=80)
    if len(mit_papers) == 0:
        print("  No matches at threshold 80. Try lowering threshold or different query.")
    else:
        for _, row in mit_papers.iterrows():
            affs = row["affiliations"]
            affs_str = affs[:100] + "..." if isinstance(affs, str) and len(affs) > 100 else affs
            print(f"  {row['title']}")
            print(f"    Score: {row['affiliation_score']} | Affiliations: {affs_str}")

    print("\n" + "=" * 60)
    print("TEST 5: get_topic_overview() — first 3 topics")
    print("=" * 60)
    overview = get_topic_overview()
    print(overview.head(3).to_string())

    print("\n" + "=" * 60)
    print("All engine functions exercised. Ready for Streamlit integration.")
    print("=" * 60)