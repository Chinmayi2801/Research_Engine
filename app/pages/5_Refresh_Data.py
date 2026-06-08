"""Refresh Data - corpus status, file timestamps, and refresh pipeline information."""

import os
import sys
from datetime import datetime

# bootstrap
_search = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    _candidate = os.path.join(_search, "src")
    if os.path.exists(os.path.join(_candidate, "engine.py")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break
    _search = os.path.dirname(_search)

import streamlit as st
import pandas as pd


st.set_page_config(page_title="Refresh Data - Re-Search Engine", page_icon="🔬", layout="wide")

st.title("Refresh Data")
st.markdown(
    "Corpus status, file freshness, and information about the data refresh pipeline."
)


@st.cache_data(show_spinner=False)
def load_corpus_status():
    df = pd.read_csv("../models/papers_with_topics.csv")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    topic_queries = []
    if "topic_query" in df.columns:
        topic_queries = sorted(df["topic_query"].dropna().unique().tolist())

    return {
        "total_papers": len(df),
        "total_topics": int(df[df["topic"] != -1]["topic"].nunique()),
        "papers_per_topic": df[df["topic"] != -1]["topic"].value_counts().sort_index(),
        "date_min": df["published_date"].min(),
        "date_max": df["published_date"].max(),
        "topic_queries": topic_queries,
    }


def file_age(path):
    if not os.path.exists(path):
        return None, False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    age = datetime.now() - mtime
    if age.days > 0:
        return f"{age.days} day{'s' if age.days != 1 else ''} ago", True
    hours = age.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} ago", True
    minutes = max(age.seconds // 60, 1)
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago", True


# ----------------------------------------------------------------------------
# SECTION 1: CORPUS STATUS
# ----------------------------------------------------------------------------

st.divider()
st.subheader("Current corpus")

try:
    status = load_corpus_status()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total papers", f"{status['total_papers']:,}")
    col2.metric("Topic clusters", status["total_topics"])
    col3.metric(
        "Earliest paper",
        status["date_min"].strftime("%Y-%m-%d") if pd.notna(status["date_min"]) else "N/A",
    )
    col4.metric(
        "Latest paper",
        status["date_max"].strftime("%Y-%m-%d") if pd.notna(status["date_max"]) else "N/A",
    )

    if status["topic_queries"]:
        st.markdown("**Topics covered in this fetch:**")
        st.markdown(", ".join(f"`{q}`" for q in status["topic_queries"]))

    st.markdown("**Papers per topic cluster**")
    chart_df = pd.DataFrame({
        "Topic ID": status["papers_per_topic"].index.astype(str),
        "Papers": status["papers_per_topic"].values,
    })
    st.bar_chart(chart_df, x="Topic ID", y="Papers")

except Exception as e:
    st.error(f"Could not load corpus status: {e}")


# ----------------------------------------------------------------------------
# SECTION 2: FILE TIMESTAMPS
# ----------------------------------------------------------------------------

st.divider()
st.subheader("Data artifact freshness")
st.caption("When was each pipeline artifact last updated?")

files = [
    ("../data/master_papers.csv", "Master papers CSV"),
    ("../models/paper_embeddings.npy", "Paper embeddings (FAISS input)"),
    ("../models/faiss_index.bin", "FAISS index"),
    ("../models/bertopic_model", "BERTopic model"),
    ("../models/papers_with_topics.csv", "Papers with topic assignments"),
    ("../models/pais_lgb_model.pkl", "PAIS LightGBM model"),
    ("../models/papers_with_pais.csv", "Papers with PAIS scores"),
    ("../models/rag_cache.json", "RAG summary cache"),
]

for path, name in files:
    age_str, exists = file_age(path)
    if exists:
        st.markdown(f"- **{name}** — updated {age_str} ✓")
    else:
        st.markdown(f"- **{name}** — not found ✗")


# ----------------------------------------------------------------------------
# SECTION 3: REFRESH PIPELINE INFO
# ----------------------------------------------------------------------------

st.divider()
st.subheader("Refresh pipeline")

st.markdown(
    """
The data refresh pipeline runs as a sequence of scripts. In production this would
be triggered on a schedule (daily or weekly). For this demo, the actual refresh
is run from the command line — the UI shows status only.

**Sequence to refresh the entire pipeline:**
```bash
cd src
python data_pipeline.py        # fetch new papers from arXiv + Semantic Scholar
python embeddings.py           # re-embed
python topic_modeling.py       # re-cluster topics with BERTopic
python search_engine.py        # rebuild FAISS index
python train_pais_model.py     # re-apply PAIS to new papers
```

Total runtime: 10-30 minutes depending on the number of new papers and
arXiv API throttling.
"""
)

with st.expander("Why isn't there a 'Run pipeline' button?"):
    st.markdown(
        """
A button-triggered refresh has two problems:

1. **Runtime** — fetching, embedding, clustering, and scoring takes 10-30 minutes.
   A Streamlit demo session blocks on this, which is bad UX.

2. **External API reliability** — arXiv has been throttling requests since early
   2026. Running the pipeline from the UI risks failure mid-fetch.

A production deployment would solve this with a scheduled background job
(e.g. cron + a worker queue) and have the Streamlit app consume artifacts
that are always present and up to date. This admin page would then show
"last refresh successful" rather than running it directly.

This is documented in the blackbook as a future-work item.
"""
    )

if st.button("Demonstrate refresh (no-op)", help="In production this would trigger the pipeline."):
    with st.spinner("In production, this would now run the pipeline in the background..."):
        import time
        time.sleep(2)
    st.success("Demo complete — no actual refresh was performed.")
    st.info("To actually refresh the corpus, run the commands shown above from the terminal.")