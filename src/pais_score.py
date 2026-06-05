import pandas as pd
import numpy as np
from datetime import datetime
import os


def compute_pais_scores(papers_path="../data/master_papers.csv",
                       save_path="../models/papers_with_pais.csv",
                       weights=None):
    """
    Computes the Predictive Academic Influence Score (PAIS) for every paper.
    
    PAIS combines 8 normalized features into a single influence score:
      - citation_count
      - influential_citation_count
      - citation_velocity (citations per month)
      - mean_h_index (mean h-index across authors)
      - max_h_index (strongest author)
      - reference_count
      - venue_score (top venue / mid / arxiv only)
      - recency
    """
    if weights is None:
        weights = {
            "citations": 0.20,
            "influential": 0.20,
            "velocity": 0.15,
            "mean_h": 0.10,
            "max_h": 0.10,
            "references": 0.05,
            "venue": 0.10,
            "recency": 0.10
        }
    
    print(f"Loading papers from {papers_path}")
    df = pd.read_csv(papers_path)
    print(f"Loaded {len(df)} papers")
    
    # ensure numeric columns
    for col in ["citation_count", "influential_citation_count",
                "reference_count", "mean_h_index", "max_h_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # convert dates
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    df["days_since_published"] = (today - df["published_date"]).dt.days
    df["days_since_published"] = df["days_since_published"].fillna(
        df["days_since_published"].max()
    ).clip(lower=1)  # prevent division by zero
    
    # citation velocity: citations per month
    df["citation_velocity"] = df["citation_count"] / (df["days_since_published"] / 30)
    
    # venue tier scoring
    top_venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP",
                  "AAAI", "IJCAI", "Nature", "Science", "JMLR"]
    mid_venues = ["IEEE", "ACM", "Springer", "Elsevier"]
    
    def venue_score(venue):
        if not isinstance(venue, str) or venue.strip() == "":
            return 0.0
        venue_lower = venue.lower()
        for v in top_venues:
            if v.lower() in venue_lower:
                return 1.0
        for v in mid_venues:
            if v.lower() in venue_lower:
                return 0.5
        return 0.3  # has a venue but not in our lists
    
    df["venue_score"] = df["venue"].apply(venue_score)
    
    # min-max normalize each component
    def min_max_normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    df["norm_citations"] = min_max_normalize(df["citation_count"])
    df["norm_influential"] = min_max_normalize(df["influential_citation_count"])
    df["norm_velocity"] = min_max_normalize(df["citation_velocity"])
    df["norm_mean_h"] = min_max_normalize(df["mean_h_index"])
    df["norm_max_h"] = min_max_normalize(df["max_h_index"])
    df["norm_references"] = min_max_normalize(df["reference_count"])
    df["norm_recency"] = 1 - min_max_normalize(df["days_since_published"])
    # venue_score is already 0-1, no normalization needed
    
    # compute PAIS
    df["pais_score"] = (
        weights["citations"] * df["norm_citations"]
        + weights["influential"] * df["norm_influential"]
        + weights["velocity"] * df["norm_velocity"]
        + weights["mean_h"] * df["norm_mean_h"]
        + weights["max_h"] * df["norm_max_h"]
        + weights["references"] * df["norm_references"]
        + weights["venue"] * df["venue_score"]
        + weights["recency"] * df["norm_recency"]
    )
    
    # save
    os.makedirs("../models", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved papers with PAIS scores to {save_path}")
    
    # statistics
    print("\n--- PAIS Score Statistics ---")
    print(df["pais_score"].describe())
    
    # top 10
    print("\n--- Top 10 papers by PAIS ---")
    top10 = df.nlargest(10, "pais_score")[
        ["title", "mean_h_index", "max_h_index", "reference_count",
         "venue", "citation_count", "pais_score"]
    ]
    for i, row in top10.iterrows():
        print(f"\n  {row['title']}")
        print(f"  Mean H: {row['mean_h_index']:.1f} | Max H: {row['max_h_index']:.0f} "
              f"| Refs: {row['reference_count']:.0f} | Citations: {row['citation_count']:.0f}")
        print(f"  Venue: {row['venue'] if row['venue'] else 'None'}")
        print(f"  PAIS: {row['pais_score']:.3f}")
    
    return df


if __name__ == "__main__":
    df = compute_pais_scores()