"""Topic Explorer — placeholder, coming on Day 18."""

import os
import sys

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

st.set_page_config(page_title="Topic Explorer — Re-Search Engine", page_icon="🔬", layout="wide")

st.title("Topic Explorer")
st.info("Coming on Day 18.")
st.markdown("**Planned behavior:** Browse all topic clusters with top words and representative papers. Click into a topic to drill down.")