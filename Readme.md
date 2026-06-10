# Re-Search Engine

A research discovery platform for STEM papers. Semantic search, topic clustering, predictive influence scoring, and LLM-generated topic summaries — in one Streamlit app.

---

## Features

| Feature | Description |
|---|---|
| **Semantic Search** | FAISS cosine-similarity search re-ranked by hybrid of similarity + PAIS |
| **Related Papers** | Top-K most similar papers for any selected paper |
| **Topic Explorer** | BERTopic clusters with drill-down and similar-topic navigation |
| **RAG Topic Summary** | Groq Llama-3.3-70b synthesis across retrieved papers |
| **Institution Filter** | Fuzzy affiliation matching via rapidfuzz |
| **Refresh Data** | Corpus status, pipeline artifact freshness |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Chinmayi2801/Research_Engine.git
cd Research_Engine
```

### 2. Create the conda environment
```bash
conda create -n research_engine python=3.12
conda activate research_engine
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the project root:
```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Get keys from:
- Semantic Scholar: https://www.semanticscholar.org/product/api
- Groq: https://console.groq.com

### 5. Run the data pipeline
```bash
cd src
python data_pipeline.py          # fetch papers from arXiv + Semantic Scholar
python embeddings.py             # generate sentence embeddings
python topic_modeling.py         # fit BERTopic clusters
python search_engine.py          # build FAISS index
python fetch_historical_ss.py all  # fetch 2019-2020 training data for PAIS
python train_pais_model.py       # train LightGBM PAIS model + score papers
```

### 6. Launch the app
```bash
cd ..
streamlit run app/Home.py
```

Open `http://localhost:8501` in your browser.

---

## Architecture

```
arXiv API ──┐
            ├──► data_pipeline.py ──► master_papers.csv
Semantic    │
Scholar API─┘
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                    embeddings.py   topic_modeling.py   train_pais_model.py
                          │                │                ▼
                    paper_embeddings  bertopic_model   pais_lgb_model.pkl
                          │                │                │
                          ▼                ▼                ▼
                    search_engine.py  papers_with_topics  papers_with_pais
                          │
                          ▼
                      faiss_index.bin
                          │
                    ┌─────┴──────┐
                    ▼            ▼
               engine.py   rag_summarizer.py ──► Groq API
                    │
                    ▼
              Streamlit app (app/)
```

---

## Project structure

```
research_engine/
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Search.py
│       ├── 2_Topic_Explorer.py
│       ├── 3_RAG_Summary.py
│       ├── 4_Institution_Filter.py
│       └── 5_Refresh_Data.py
├── src/
│   ├── arxiv_fetcher.py          # arXiv API fetch
│   ├── semantic_scholar_fetcher.py  # Semantic Scholar enrichment
│   ├── data_pipeline.py          # orchestrates fetch + enrich + clean
│   ├── embeddings.py             # sentence-transformer embeddings
│   ├── topic_modeling.py         # BERTopic clustering
│   ├── search_engine.py          # FAISS index build + search
│   ├── fetch_historical_ss.py    # Semantic Scholar historical fetch (2019-2020)
│   ├── re_enrich.py              # re-enrich master_papers with correct h-index
│   ├── train_pais_model.py       # LightGBM PAIS training + scoring
│   ├── topic_trends.py           # topic exploration helpers
│   ├── rag_summarizer.py         # Groq RAG summarization with cache
│   └── engine.py                 # orchestration layer for Streamlit
├── data/                         # gitignored — generated at runtime
├── models/                       # gitignored — generated at runtime
├── notebooks/                    # exploration notebooks
├── requirements.txt
├── .env                          # gitignored — API keys
├── .gitignore
└── README.md
```

---

## Data

The corpus is fetched from arXiv and enriched via Semantic Scholar. The current build contains **507 papers** across five topics: machine learning, computer vision, NLP, reinforcement learning, and graph neural networks.

The PAIS model is trained on **527 historical papers** (2019-2020) with real citation history to avoid the cold-start problem on fresh 2026 papers.

---

## PAIS — Predictive Academic Influence Score

PAIS is a LightGBM regression model trained to predict log-transformed citation count from six intrinsic paper features:

| Feature | Description |
|---|---|
| `reference_count` | Number of references — proxy for methodological grounding |
| `mean_h_index` | Mean h-index across all authors |
| `max_h_index` | Maximum h-index across all authors |
| `venue_score` | Tiered venue prestige (NeurIPS/ICML/ACL = 1.0, IEEE/ACM = 0.5, other = 0.3) |
| `num_authors` | Number of authors |
| `abstract_length` | Character length of abstract |

**Test R² = 0.154** on a held-out 20% split of the historical training data.

Papers with no author or reference metadata (all three zero) are masked and receive no PAIS score rather than a misleading prediction.

---

## Known limitations

- **Corpus window**: all papers are from a one-week window (2026-06-01). Temporal trend analysis is excluded for this reason.
- **PAIS distribution shift**: model trained on 2019-2020 papers, applied to 2026 papers. Citation patterns may differ.
- **Venue sparsity**: Semantic Scholar does not return venue data for most arXiv preprints until formal publication. `venue_score` is constant (0.3) for ~97% of the apply set.
- **RAG hallucination risk**: the LLM occasionally force-fits tangentially-related papers into the query's narrative. Prompt engineering mitigates but does not eliminate this.
- **Affiliation coverage**: ~90% of papers have h-index and reference data; affiliation data is sparser and limits institution filter recall.

---

## License

MIT