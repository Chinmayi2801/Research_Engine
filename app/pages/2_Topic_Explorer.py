"""Topic Explorer - browse BERTopic clusters, drill into papers, navigate similar topics."""

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
from engine import get_topic_overview, get_topic_papers, get_topic_similarities


st.set_page_config(page_title="Topic Explorer - Re-Search Engine", page_icon="🔬", layout="wide")

st.title("Topic Explorer")
st.markdown(
    "Browse the corpus organized into thematic clusters discovered by BERTopic. "
    "Papers within a topic share semantic similarity. Click into any topic to see "
    "its top papers (ranked by PAIS) and conceptually related topics."
)


@st.cache_data(show_spinner=False)
def load_overview():
    return get_topic_overview()


@st.cache_data(show_spinner=False)
def load_topic_papers(topic_id, top_n=10):
    return get_topic_papers(topic_id, top_n=top_n, sort_by="pais")


@st.cache_data(show_spinner=False)
def load_all_similarities():
    return get_topic_similarities(top_pairs=500)


def similar_topics_for(topic_id, sims_df, top_n=5):
    matches = sims_df[
        (sims_df["topic_a"] == topic_id) | (sims_df["topic_b"] == topic_id)
    ]
    result = []
    for _, row in matches.iterrows():
        other = int(row["topic_b"] if row["topic_a"] == topic_id else row["topic_a"])
        result.append({"topic_id": other, "similarity": float(row["similarity"])})
    return result[:top_n]


# ----------------------------------------------------------------------------
# CALLBACK: navigate to a different topic
# ----------------------------------------------------------------------------

def jump_to_topic(topic_id):
    """Set the pending topic. Runs BEFORE the next rerun, so the selectbox
    instantiates with the new index."""
    st.session_state["pending_topic"] = topic_id


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------

try:
    overview = load_overview()
    similarities = load_all_similarities()
except Exception as e:
    st.error(f"Could not load topic data: {e}")
    st.info("Make sure topic_modeling.py has been run and the BERTopic model is saved.")
    st.stop()


# ----------------------------------------------------------------------------
# SECTION 1: OVERVIEW TABLE
# ----------------------------------------------------------------------------

st.divider()
st.subheader(f"All topics ({len(overview)})")
st.caption("Topics ordered by size (number of papers). Scroll to browse.")

display_df = overview.copy()
display_df["Top words"] = display_df["top_words"].apply(
    lambda x: ", ".join(x[:8]) if isinstance(x, list) else ""
)
display_df = display_df[["topic_id", "size", "Top words"]]
display_df.columns = ["Topic ID", "Papers", "Top words"]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=350,
)


# ----------------------------------------------------------------------------
# SECTION 2: TOPIC DETAIL
# ----------------------------------------------------------------------------

st.divider()
st.subheader("Inspect a topic")

topic_options_dict = {
    int(row["topic_id"]): f"Topic {int(row['topic_id'])} — {row['label']} ({row['size']} papers)"
    for _, row in overview.iterrows()
}
options_list = list(topic_options_dict.keys())

# initialize the pending topic on first load
if "pending_topic" not in st.session_state:
    st.session_state["pending_topic"] = options_list[0]

# compute index from pending_topic
try:
    default_idx = options_list.index(st.session_state["pending_topic"])
except ValueError:
    default_idx = 0

# NO key= on selectbox — we manage navigation via pending_topic ourselves
selected = st.selectbox(
    "Select a topic",
    options=options_list,
    format_func=lambda x: topic_options_dict[x],
    index=default_idx,
)

# keep pending_topic in sync with whatever the user manually picked
st.session_state["pending_topic"] = selected


if selected is not None:
    topic_row = overview[overview["topic_id"] == selected].iloc[0]

    # header
    st.markdown(f"### Topic {selected}: {topic_row['label']}")

    # metrics
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Papers in this topic", topic_row["size"])
    mcol2.metric("Out of total topics", len(overview))

    # top words
    top_words = topic_row["top_words"]
    if isinstance(top_words, list):
        st.markdown(f"**Top words:** {', '.join(top_words)}")

    # top papers
    st.markdown("---")
    st.markdown("**Top papers in this topic, ranked by PAIS**")

    papers = load_topic_papers(selected, top_n=10)

    if len(papers) == 0:
        st.info("No papers in this topic.")
    else:
        for _, paper in papers.iterrows():
            pcol1, pcol2 = st.columns([5, 1])
            with pcol1:
                st.markdown(f"**{paper['title']}**")
                authors = paper.get("authors", "Unknown")
                if isinstance(authors, str) and len(authors) > 150:
                    authors = authors[:150] + "..."
                st.caption(f"{authors}")
                st.caption(
                    f"arXiv: `{paper['arxiv_id']}` · Published: {paper.get('published_date', 'N/A')}"
                )
            with pcol2:
                pais = paper.get("pais_score")
                if pd.notna(pais):
                    st.metric("PAIS", f"{pais:.3f}", label_visibility="visible")
                else:
                    st.caption("PAIS: N/A")
            st.markdown("---")

    # similar topics — use on_click callback (NOT direct session_state mutation)
    st.markdown("**Most similar topics**")
    similar = similar_topics_for(selected, similarities, top_n=5)

    if len(similar) == 0:
        st.info("No similar topics found.")
    else:
        st.caption("Click any similar topic to jump to it.")
        for s in similar:
            other_row = overview[overview["topic_id"] == s["topic_id"]]
            if len(other_row) > 0:
                other_label = other_row.iloc[0]["label"]
                other_size = int(other_row.iloc[0]["size"])
                btn_label = (
                    f"Topic {s['topic_id']} — {other_label} "
                    f"({other_size} papers) · similarity {s['similarity']:.3f}"
                )
                st.button(
                    btn_label,
                    key=f"sim_btn_{s['topic_id']}",
                    on_click=jump_to_topic,
                    args=(s["topic_id"],),
                )