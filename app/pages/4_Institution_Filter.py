"""Institution Filter - find papers from specific institutions via fuzzy affiliation matching."""

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
from engine import filter_by_institution


st.set_page_config(page_title="Institution Filter - Re-Search Engine", page_icon="🔬", layout="wide")

st.title("Institution Filter")
st.markdown(
    "Find papers from specific institutions using fuzzy matching on author affiliations. "
    "Matching uses `rapidfuzz.token_set_ratio` which handles abbreviations and word-order "
    "differences (e.g. 'MIT' matches 'Massachusetts Institute of Technology')."
)
st.caption("Note: only papers where Semantic Scholar returned affiliation data can be matched.")


@st.cache_data(show_spinner=False)
def run_filter(institution, top_k, threshold):
    return filter_by_institution(institution, top_k=top_k, score_threshold=threshold)


with st.form("inst_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        institution = st.text_input(
            "Institution",
            placeholder="e.g. MIT, Stanford, Google, IIT Bombay, Microsoft Research",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.slider("Max results", 5, 50, 20)

    threshold = st.slider(
        "Match strictness", 0, 100, 80, 5,
        help="Higher = stricter match. 100=exact substring, 80=close, 60=loose."
    )

    submitted = st.form_submit_button("Filter", type="primary")


if submitted:
    if not institution.strip():
        st.warning("Please enter an institution name.")
    else:
        try:
            with st.spinner("Searching affiliations..."):
                results = run_filter(institution.strip(), top_k, threshold)
            st.session_state["filter_results"] = results
            st.session_state["filter_query"] = institution.strip()
            st.session_state["filter_threshold"] = threshold
        except Exception as e:
            st.error(f"Filter failed: {e}")
            st.session_state.pop("filter_results", None)


if "filter_results" in st.session_state:
    results = st.session_state["filter_results"]
    query = st.session_state["filter_query"]
    threshold_used = st.session_state["filter_threshold"]

    st.divider()

    if len(results) == 0:
        st.warning(
            f"No papers found with affiliations matching '{query}' at threshold {threshold_used}."
        )
        st.info("Try lowering the threshold or using a different institution name.")
    else:
        st.subheader(f"Found {len(results)} papers matching '{query}'")

        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}**")

            mcol1, mcol2 = st.columns([4, 1])
            with mcol1:
                authors = row.get("authors", "Unknown")
                if isinstance(authors, str) and len(authors) > 200:
                    authors = authors[:200] + "..."
                st.caption(f"**Authors:** {authors}")

                affs = row.get("affiliations", "")
                if isinstance(affs, str) and affs.strip():
                    aff_display = affs[:250] + ("..." if len(affs) > 250 else "")
                    st.caption(f"**Affiliations:** {aff_display}")

                pub_date = row.get("published_date", "N/A")
                topic = row.get("topic", "N/A")
                st.caption(f"arXiv: `{row['arxiv_id']}` · Published: {pub_date} · Topic: {topic}")

            with mcol2:
                st.metric("Match score", f"{int(row['affiliation_score'])}")
                pais = row.get("pais_score")
                if pd.notna(pais):
                    st.caption(f"PAIS: {pais:.3f}")
                else:
                    st.caption("PAIS: N/A")

            st.markdown("---")
else:
    st.info("Enter an institution above and click **Filter** to get started.")