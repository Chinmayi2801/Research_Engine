"""Search — semantic search with PAIS re-ranking, expandable paper detail cards."""

import os
import sys

# bootstrap: locate src/, add to path, chdir for relative imports in engine
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
from engine import search, get_paper_details, get_related_papers


st.set_page_config(page_title="Search — Re-Search Engine", page_icon="🔬", layout="wide")

st.title("Semantic Search")
st.markdown(
    "Search the corpus by meaning, not just keywords. Results are re-ranked by a "
    "hybrid of FAISS similarity and predictive influence (PAIS)."
)


# ----------------------------------------------------------------------------
# CACHED WRAPPERS — Streamlit reruns the script on each interaction; caching
# avoids re-calling FAISS / pandas / disk reads.
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_search(query, k, sim_weight, pais_weight):
    return search(query, top_k=k, similarity_weight=sim_weight, pais_weight=pais_weight)


@st.cache_data(show_spinner=False)
def run_details(arxiv_id):
    return get_paper_details(arxiv_id)


@st.cache_data(show_spinner=False)
def run_related(arxiv_id, k):
    return get_related_papers(arxiv_id, top_k=k)


# ----------------------------------------------------------------------------
# SEARCH FORM
# ----------------------------------------------------------------------------

with st.form("search_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. graph neural networks for drug discovery",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.slider("Results", min_value=5, max_value=20, value=10)

    with st.expander("Advanced — adjust ranking weights"):
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            sim_weight = st.slider("Similarity weight", 0.0, 1.0, 0.6, 0.1)
        with wcol2:
            pais_weight = st.slider("PAIS weight", 0.0, 1.0, 0.4, 0.1)
        st.caption(
            f"Hybrid score = {sim_weight:.1f} × similarity + {pais_weight:.1f} × PAIS"
        )

    submitted = st.form_submit_button("Search", type="primary")


# ----------------------------------------------------------------------------
# HANDLE SUBMISSION
# ----------------------------------------------------------------------------

if submitted:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            with st.spinner("Searching..."):
                results = run_search(query.strip(), top_k, sim_weight, pais_weight)
            st.session_state["search_results"] = results
            st.session_state["search_query"] = query.strip()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state.pop("search_results", None)


# ----------------------------------------------------------------------------
# RESULTS DISPLAY
# ----------------------------------------------------------------------------

if "search_results" in st.session_state:
    results = st.session_state["search_results"]
    query_shown = st.session_state["search_query"]

    if len(results) == 0:
        st.info(f"No results found for: \"{query_shown}\"")
    else:
        st.divider()
        st.subheader(f"Top {len(results)} results for: \"{query_shown}\"")
        st.caption("Click any result to expand details and see related papers.")

        for i, row in results.iterrows():
            rank = i + 1
            title = row["title"]
            hybrid = row["hybrid_score"]

            with st.expander(f"**{rank}.** {title}  —  hybrid {hybrid:.3f}"):
                # ---- score metrics ----
                mcol1, mcol2, mcol3 = st.columns(3)
                sim = row.get("similarity", 0)
                pais = row.get("pais_score")
                pais_str = f"{pais:.3f}" if pd.notna(pais) else "N/A"

                mcol1.metric("Similarity", f"{sim:.3f}")
                mcol2.metric("PAIS", pais_str)
                mcol3.metric("Hybrid score", f"{hybrid:.3f}")

                # ---- paper details ----
                details = run_details(row["arxiv_id"])

                if details:
                    icol1, icol2 = st.columns([3, 1])
                    with icol1:
                        st.markdown(f"**Authors:** {details.get('authors', 'N/A')}")
                    with icol2:
                        st.markdown(f"**Published:** {details.get('published_date', 'N/A')}")
                    st.markdown(f"**arXiv ID:** `{details.get('arxiv_id', 'N/A')}`  •  **Topic:** {details.get('topic', 'N/A')}")

                    abstract = details.get("abstract")
                    if isinstance(abstract, str) and abstract.strip():
                        st.markdown("**Abstract**")
                        st.markdown(abstract)

                # ---- related papers ----
                st.markdown("---")
                st.markdown("**Top 5 related papers**")
                try:
                    related = run_related(row["arxiv_id"], 5)
                    if len(related) > 0:
                        for _, rel in related.iterrows():
                            rel_sim = rel.get("similarity", 0)
                            rel_pais = rel.get("pais_score")
                            rel_pais_str = (
                                f"PAIS {rel_pais:.3f}" if pd.notna(rel_pais) else "PAIS N/A"
                            )
                            st.markdown(
                                f"- *{rel['title']}*  \n"
                                f"  similarity {rel_sim:.3f}  ·  {rel_pais_str}"
                            )
                    else:
                        st.info("No related papers found.")
                except Exception as e:
                    st.error(f"Could not load related papers: {e}")

else:
    st.info("Enter a query above and click **Search** to get started.")