import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


FEATURES = [
    "reference_count",
    "mean_h_index",
    "max_h_index",
    "venue_score",
    "num_authors",
    "abstract_length",
]

TARGET = "citation_count"


def venue_score(venue):
    """Tier scoring for venues."""
    if not isinstance(venue, str) or venue.strip() == "":
        return 0.0
    venue_lower = venue.lower()
    top_venues = ["neurips", "icml", "iclr", "cvpr", "acl", "emnlp",
                  "aaai", "ijcai", "nature", "science", "jmlr"]
    mid_venues = ["ieee", "acm", "springer", "elsevier"]
    for v in top_venues:
        if v in venue_lower:
            return 1.0
    for v in mid_venues:
        if v in venue_lower:
            return 0.5
    return 0.3


def prepare_features(df):
    """Builds the feature matrix from the master CSV."""
    df = df.copy()

    # ensure numeric columns
    for col in ["citation_count", "influential_citation_count",
                "reference_count", "mean_h_index", "max_h_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # dates
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    df["days_since_published"] = (today - df["published_date"]).dt.days
    df["days_since_published"] = df["days_since_published"].fillna(
        df["days_since_published"].max()
    ).clip(lower=1)

    # venue score
    df["venue_score"] = df["venue"].apply(venue_score)

    # other derived features
    df["num_authors"] = df["authors"].fillna("").apply(
        lambda x: len(x.split(",")) if x else 0
    )
    df["title_length"] = df["title"].fillna("").apply(len)
    df["abstract_length"] = df["abstract"].fillna("").apply(len)

    return df


def train_pais_model(papers_path, model_save_path="../models/pais_lgb_model.pkl"):
    """
    Trains a LightGBM regressor to predict citation count from intrinsic features.
    """
    print(f"Loading training data from {papers_path}")
    df = pd.read_csv(papers_path)
    print(f"Loaded {len(df)} papers")

    df = prepare_features(df)

    # check if we have any non-zero targets
    nonzero = (df[TARGET] > 0).sum()
    print(f"Papers with non-zero citation count: {nonzero}/{len(df)}")

    if nonzero < 20:
        print("\nWARNING: Not enough labeled data (need 20+ papers with citations).")
        print("Model will train but predictions won't be meaningful.")
        print("Once historical data is fetched, retrain by running this script again.")

    X = df[FEATURES].values
    y = df[TARGET].values

    # log-transform target — citation counts are heavily skewed
    y_log = np.log1p(y)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # train LightGBM
    print("\nTraining LightGBM regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest RMSE (log scale): {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")

    # feature importances
    print("\n--- Feature importances ---")
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance_df.to_string(index=False))

    # save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

    return model


def apply_pais_model(papers_path,
                    model_path="../models/pais_lgb_model.pkl",
                    save_path="../models/papers_with_pais.csv"):
    """
    Loads a trained model and applies it to score papers in the given CSV.
    Papers with no real feature signal (no refs, no h-index) are masked
    to NaN so they don't get a misleading PAIS score.
    """
    print(f"Loading papers from {papers_path}")
    df = pd.read_csv(papers_path)

    df = prepare_features(df)

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    X = df[FEATURES].values

    # predict in log space then invert
    log_predictions = model.predict(X)
    df["pais_predicted_citations"] = np.maximum(np.expm1(log_predictions), 0)

    # mask papers with no real feature signal
    no_signal_mask = (
        (df["reference_count"] == 0) &
        (df["mean_h_index"] == 0) &
        (df["max_h_index"] == 0)
    )
    df.loc[no_signal_mask, "pais_predicted_citations"] = np.nan
    print(f"Masked {no_signal_mask.sum()} papers with no feature signal")

    # normalize to 0-1 (NaN values stay NaN; .min() and .max() skip NaN)
    min_p = df["pais_predicted_citations"].min()
    max_p = df["pais_predicted_citations"].max()
    if max_p > min_p:
        df["pais_score"] = (df["pais_predicted_citations"] - min_p) / (max_p - min_p)
    else:
        df["pais_score"] = 0.5

    df.to_csv(save_path, index=False)
    print(f"\nSaved scored papers to {save_path}")

    # show top 10 — only papers with valid (non-NaN) PAIS scores
    print("\n--- Top 10 papers by predicted PAIS ---")
    valid_df = df.dropna(subset=["pais_score"])
    top10 = valid_df.nlargest(10, "pais_score")[
        ["title", "mean_h_index", "max_h_index", "reference_count",
         "venue", "pais_predicted_citations", "pais_score"]
    ]
    for i, row in top10.iterrows():
        print(f"\n  {row['title']}")
        print(f"  Predicted citations: {row['pais_predicted_citations']:.1f} | PAIS: {row['pais_score']:.3f}")
        print(f"  Mean H: {row['mean_h_index']:.1f} | Max H: {row['max_h_index']:.0f} | Refs: {row['reference_count']:.0f}")

    return df


if __name__ == "__main__":
    # train on historical data — commented out, model already trained and saved
    # train_pais_model("../data/historical_master.csv")

    # apply trained model to score the 2026 papers
    apply_pais_model("../data/master_papers.csv")