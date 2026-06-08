"""RAG Summary - LLM-generated synthesis of a research topic from retrieved papers."""

import os
import sys

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
from engine import get_topic_summary


st.set_page_config(page_title="RAG Summary - Re-Search Engine", page_icon="🔬", layout="wide")

st.title("RAG Topic Summary")
st.markdown(
    "Enter a research area. The system retrieves the most relevant papers via FAISS "
    "and generates a synthesized summary using Groq's Llama-3.3-70b model. Results "
    "are cached so repeated queries return instantly."
)


@st.cache_data(show_spinner=False)
def generate_summary(query, top_k):
    return get_topic_summary(query, top_k=top_k)


with st.form("rag_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Topic query",
            placeholder="e.g. graph neural networks for drug discovery",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.slider("Papers in context", 5, 15, 8)
    submitted = st.form_submit_button("Generate summary", type="primary")


if submitted:
    if not query.strip():
        st.warning("Please enter a topic query.")
    else:
        try:
            with st.spinner("Retrieving papers and generating summary..."):
                result = generate_summary(query.strip(), top_k)
            st.session_state["rag_result"] = result
        except Exception as e:
            st.error(f"Summary generation failed: {e}")
            st.session_state.pop("rag_result", None)


if "rag_result" in st.session_state:
    result = st.session_state["rag_result"]

    st.divider()

    # cache indicator
    if result.get("from_cache"):
        st.success(f"⚡ Cached result returned instantly. Query: \"{result['query']}\"")
    else:
        st.info(f"✨ Freshly generated via Groq. Query: \"{result['query']}\"")

    # summary
    st.markdown("### Summary")
    st.markdown(result["summary"])

    # papers used
    st.markdown("---")
    st.markdown(f"### Papers used as context ({len(result['papers_used'])})")
    st.caption("These are the papers retrieved by FAISS and fed to the LLM.")

    for i, p in result["papers_used"].iterrows():
        pcol1, pcol2 = st.columns([5, 1])
        with pcol1:
            st.markdown(f"**{i + 1}. {p['title']}**")
            authors = p.get("authors", "Unknown")
            if isinstance(authors, str) and len(authors) > 200:
                authors = authors[:200] + "..."
            st.caption(f"{authors}")
            st.caption(f"arXiv: `{p['arxiv_id']}`")
        with pcol2:
            sim = p.get("similarity", 0)
            st.metric("Similarity", f"{sim:.3f}")
        st.markdown("---")
else:
    st.info("Enter a topic query above and click **Generate summary** to get started.")