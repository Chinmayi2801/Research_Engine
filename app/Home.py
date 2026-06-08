"""
Re-Search Engine — Home page.

Run from the project root:
    streamlit run app/Home.py
"""

import os
import sys

# Bootstrap: locate src/, add to sys.path, set CWD so existing modules' relative paths work.
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


st.set_page_config(
    page_title="Re-Search Engine",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data
def load_corpus_stats():
    df = pd.read_csv("../models/papers_with_topics.csv")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    return {
        "total_papers": len(df),
        "total_topics": int(df[df["topic"] != -1]["topic"].nunique()),
        "date_min": df["published_date"].min().strftime("%Y-%m-%d") if df["published_date"].notna().any() else "N/A",
        "date_max": df["published_date"].max().strftime("%Y-%m-%d") if df["published_date"].notna().any() else "N/A",
    }


st.title("Re-Search Engine")
st.markdown(
    "A research discovery platform for STEM papers. Search semantically, "
    "see related work, explore topic clusters, get AI-generated summaries, "
    "and filter by author affiliation."
)

st.divider()
st.subheader("Corpus overview")

try:
    stats = load_corpus_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers indexed", f"{stats['total_papers']:,}")
    col2.metric("Topic clusters", f"{stats['total_topics']}")
    col3.metric("Earliest paper", stats["date_min"])
    col4.metric("Latest paper", stats["date_max"])
except Exception as e:
    st.error(f"Could not load corpus data: {e}")
    st.info("Make sure you've run the data pipeline, embeddings, and topic modeling first.")

st.divider()
st.subheader("Features")

st.markdown(
    """
Use the **sidebar** on the left to navigate:

- **Search** — semantic search across the corpus, re-ranked by hybrid of similarity and predictive influence (PAIS).
- **Topic Explorer** — browse thematic clusters, drill into papers, see related topics.
- **RAG Summary** — get an LLM-generated synthesis of a research topic across retrieved papers.
- **Institution Filter** — find papers from specific institutions via fuzzy author-affiliation matching.
- **Refresh Data** — re-run the data pipeline to fetch new papers.
"""
)

st.divider()
st.caption(
    "Built with FAISS for semantic search, BERTopic for topic clustering, "
    "LightGBM for predictive influence scoring (PAIS), Groq's Llama-3.3-70b for summarization, "
    "and Streamlit for the interface."
)