"""
Day 13: Topic exploration and trend analysis on top of BERTopic clusters.

Functions:
1. get_topic_overview()              - all topics with size, top words, top papers
2. get_topic_papers()                 - papers in a topic, sorted by PAIS or date
3. get_topic_similarities()           - most similar topic pairs by embedding centroids
4. get_temporal_distribution()        - papers-per-week across the corpus
5. get_temporal_distribution_by_topic - papers-per-week split by top-N topics

Note on temporal range:
The dataset spans approximately May-June 2026 (~1 month). Strong year-over-year
trend analysis would require a longer corpus window. The temporal functions still
provide a small chart for the Streamlit UI as a within-window view.
"""

import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity


PAPERS_WITH_TOPICS = "../models/papers_with_topics.csv"
PAPERS_WITH_PAIS = "../models/papers_with_pais.csv"
EMBEDDINGS_PATH = "../models/paper_embeddings.npy"
BERTOPIC_PATH = "../models/bertopic_model"


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------

def _load_papers_with_topics():
    """Load papers_with_topics.csv. Row order matches paper_embeddings.npy."""
    return pd.read_csv(PAPERS_WITH_TOPICS)


def _load_with_pais_join(topics_df):
    """
    Join topics_df with PAIS scores by arxiv_id.
    Returns topics_df unchanged if PAIS file is missing.
    """
    if not os.path.exists(PAPERS_WITH_PAIS):
        print(f"WARNING: {PAPERS_WITH_PAIS} not found. PAIS data unavailable.")
        return topics_df.copy()

    pais_df = pd.read_csv(PAPERS_WITH_PAIS)
    pais_cols = ["arxiv_id", "pais_score", "pais_predicted_citations"]
    pais_subset = pais_df[[c for c in pais_cols if c in pais_df.columns]]
    return topics_df.merge(pais_subset, on="arxiv_id", how="left")


# ----------------------------------------------------------------------------
# 1. TOPIC OVERVIEW
# ----------------------------------------------------------------------------

def get_topic_overview(top_n_words=8, top_n_papers=3):
    """
    Returns a DataFrame of all topics with topic_id, size, label, top_words, top_papers.
    Sorted by topic size descending. Topic -1 (outliers) is excluded.
    """
    topics_df = _load_papers_with_topics()
    df = _load_with_pais_join(topics_df)
    topic_model = BERTopic.load(BERTOPIC_PATH)

    rows = []
    for topic_id in sorted(df["topic"].unique()):
        if topic_id == -1:
            continue

        topic_papers = df[df["topic"] == topic_id]
        size = len(topic_papers)

        words = topic_model.get_topic(topic_id)
        word_list = [w for w, _ in words[:top_n_words]] if words else []
        label = ", ".join(word_list[:3])

        # rank top papers by PAIS if available, else by row order
        if "pais_score" in topic_papers.columns and topic_papers["pais_score"].notna().any():
            top_papers = topic_papers.dropna(subset=["pais_score"]) \
                                     .nlargest(top_n_papers, "pais_score")
        else:
            top_papers = topic_papers.head(top_n_papers)

        rows.append({
            "topic_id": int(topic_id),
            "size": size,
            "label": label,
            "top_words": word_list,
            "top_papers": top_papers["title"].tolist(),
        })

    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. TOPIC -> PAPERS
# ----------------------------------------------------------------------------

def get_topic_papers(topic_id, top_n=10, sort_by="pais"):
    """
    Returns papers in a topic.

    Args:
        topic_id: BERTopic topic ID
        top_n: number of papers to return
        sort_by: 'pais' (default) or 'date'
    """
    topics_df = _load_papers_with_topics()
    df = _load_with_pais_join(topics_df)

    topic_papers = df[df["topic"] == topic_id].copy()
    if len(topic_papers) == 0:
        return topic_papers

    if sort_by == "pais" and "pais_score" in topic_papers.columns:
        topic_papers = topic_papers.sort_values(
            "pais_score", ascending=False, na_position="last"
        )
    elif sort_by == "date":
        topic_papers["published_date"] = pd.to_datetime(
            topic_papers["published_date"], errors="coerce"
        )
        topic_papers = topic_papers.sort_values("published_date", ascending=False)

    cols = ["arxiv_id", "title", "authors", "published_date", "topic"]
    if "pais_score" in topic_papers.columns:
        cols.append("pais_score")

    return topic_papers[cols].head(top_n).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. TOPIC SIMILARITY
# ----------------------------------------------------------------------------

def get_topic_similarities(top_pairs=10):
    """
    Computes pairwise cosine similarity between topic centroids
    (mean of paper embeddings in each topic). Returns top-N most similar pairs.
    """
    topics_df = _load_papers_with_topics()
    embeddings = np.load(EMBEDDINGS_PATH)

    assert len(embeddings) == len(topics_df), (
        f"Embedding row count ({len(embeddings)}) does not match "
        f"papers_with_topics row count ({len(topics_df)}). Re-run topic_modeling.py."
    )

    topic_ids = sorted([t for t in topics_df["topic"].unique() if t != -1])

    centroids = []
    for topic_id in topic_ids:
        mask = (topics_df["topic"] == topic_id).values
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)

    sim_matrix = cosine_similarity(centroids)

    pairs = []
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            pairs.append({
                "topic_a": int(topic_ids[i]),
                "topic_b": int(topic_ids[j]),
                "similarity": float(sim_matrix[i, j]),
            })

    pairs_df = pd.DataFrame(pairs).sort_values(
        "similarity", ascending=False
    ).reset_index(drop=True)
    return pairs_df.head(top_pairs)


# ----------------------------------------------------------------------------
# 4. TEMPORAL DISTRIBUTION
# ----------------------------------------------------------------------------

def get_temporal_distribution(freq="W"):
    """
    Papers-per-time-bin (default weekly) across the corpus.
    Limited by the ~1-month dataset window.
    """
    df = _load_papers_with_topics()
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df = df.dropna(subset=["published_date"])

    df["time_bin"] = df["published_date"].dt.to_period(freq)
    bin_counts = df.groupby("time_bin").size().reset_index(name="paper_count")
    bin_counts["time_bin"] = bin_counts["time_bin"].astype(str)
    return bin_counts


def get_temporal_distribution_by_topic(top_n_topics=10, freq="W"):
    """
    Papers per time bin split by topic. Returns wide format: time bins x topics.
    Limited to top-N topics by size.
    """
    df = _load_papers_with_topics()
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df = df.dropna(subset=["published_date"])

    topic_sizes = df[df["topic"] != -1]["topic"].value_counts()
    top_topics = topic_sizes.head(top_n_topics).index.tolist()
    df = df[df["topic"].isin(top_topics)]

    df["time_bin"] = df["published_date"].dt.to_period(freq)
    pivot = df.pivot_table(
        index="time_bin",
        columns="topic",
        values="arxiv_id",
        aggfunc="count",
        fill_value=0,
    )
    pivot.index = pivot.index.astype(str)
    return pivot


# ----------------------------------------------------------------------------
# TEST
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Topic Overview (top 10 by size) ===")
    overview = get_topic_overview()
    print(overview.head(10).to_string())
    print(f"\nTotal topics: {len(overview)}")

    largest_topic_id = int(overview.iloc[0]["topic_id"])

    print(f"\n=== Top 5 papers in largest topic ({largest_topic_id}) ===")
    print(get_topic_papers(largest_topic_id, top_n=5).to_string())

    print("\n=== Top 10 most similar topic pairs ===")
    print(get_topic_similarities(top_pairs=10).to_string())

    print("\n=== Papers per week (all topics) ===")
    print(get_temporal_distribution().to_string())

    print("\n=== Papers per week by top-5 topics ===")
    print(get_temporal_distribution_by_topic(top_n_topics=5).to_string())