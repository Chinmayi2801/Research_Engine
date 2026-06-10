"""
RAG-based topic summarization using Groq's LLM API.

Day 10: Initial RAG pipeline.
Day 11: Prompt tuning - honesty bucketing, citation suppression, no boilerplate.
Day 12: JSON-backed persistent cache to avoid re-hitting Groq on repeated queries.

Workflow:
1. User provides a query
2. Check cache - return cached summary if query was seen before
3. Otherwise: FAISS retrieves top-N most semantically relevant papers
4. Paper metadata + abstracts are formatted into a structured prompt
5. Groq's LLM generates a coherent summary
6. Result is written to cache before returning
"""

import os
import json
import hashlib
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from search_engine import search_papers

# load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

CACHE_PATH = "../models/rag_cache.json"


# ----------------------------------------------------------------------------
# CACHE HELPERS
# ----------------------------------------------------------------------------

def _cache_key(query, top_k, temperature):
    """Generate a deterministic hash key for a (query, top_k, temperature) tuple."""
    payload = f"{query.lower().strip()}|{top_k}|{temperature}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cache():
    """Load the cache dict from disk. Returns empty dict if missing or corrupt."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Cache file unreadable ({e}). Starting fresh.")
        return {}


def _save_cache(cache):
    """Persist the cache dict to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def clear_cache():
    """Wipe the cache. Useful during development."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print(f"Cache cleared: {CACHE_PATH}")
    else:
        print("No cache file to clear.")


def cache_stats():
    """Print number of cached entries and the queries they were generated from."""
    cache = _load_cache()
    print(f"\nCache: {len(cache)} entries")
    for key, entry in cache.items():
        print(f"  [{key[:8]}] '{entry['query']}' (cached {entry.get('cached_at', 'unknown')})")


# ----------------------------------------------------------------------------
# RETRIEVAL HELPERS
# ----------------------------------------------------------------------------

def get_papers_with_abstracts(query, top_k=10,
                              papers_path="../models/papers_with_topics.csv"):
    """
    Retrieves top-k papers for a query via FAISS, then joins with the full
    papers CSV to include abstracts (which search_papers does not return).
    """
    search_results = search_papers(query, top_k=top_k)
    full_df = pd.read_csv(papers_path)

    merged = full_df[full_df["arxiv_id"].isin(search_results["arxiv_id"])].copy()
    merged = merged.merge(
        search_results[["arxiv_id", "similarity"]],
        on="arxiv_id"
    )
    merged = merged.sort_values("similarity", ascending=False).reset_index(drop=True)
    return merged


def format_papers_for_prompt(papers_df):
    """Format papers into a structured context block for the LLM prompt."""
    blocks = []
    for i, row in papers_df.iterrows():
        title = row.get("title") or "(no title)"
        authors = row.get("authors") or "(unknown)"
        abstract = row.get("abstract") or "(no abstract)"
        citations = row.get("citation_count", 0)

        if isinstance(authors, str) and len(authors) > 200:
            authors = authors[:200] + "..."
        if isinstance(abstract, str) and len(abstract) > 1500:
            abstract = abstract[:1500] + "..."

        block = (
            f"[Paper {i + 1}]\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Citation count: {citations}\n"
            f"Abstract: {abstract}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


# ----------------------------------------------------------------------------
# MAIN SUMMARIZATION FUNCTION
# ----------------------------------------------------------------------------

def summarize_topic(query, top_k=8, temperature=0.3, max_tokens=800,
                    use_cache=True, force_refresh=False):
    """
    Generates a RAG-based summary of a research topic.

    Args:
        query: search string
        top_k: number of papers to retrieve and include in the prompt
        temperature: LLM sampling temperature (lower = more focused)
        max_tokens: max tokens in the generated summary
        use_cache: read from and write to the on-disk cache
        force_refresh: bypass cache read but still write to it

    Returns:
        dict with 'query', 'summary', 'papers_used', 'from_cache'
    """
    key = _cache_key(query, top_k, temperature)

    # check cache
    if use_cache and not force_refresh:
        cache = _load_cache()
        if key in cache:
            cached = cache[key]
            print(f"Cache HIT for '{query}' (cached {cached.get('cached_at')})")
            papers_used = pd.DataFrame(cached["papers_used"])
            return {
                "query": cached["query"],
                "summary": cached["summary"],
                "papers_used": papers_used,
                "from_cache": True,
            }
        print(f"Cache MISS for '{query}', calling API...")
    elif force_refresh:
        print(f"Force refresh requested for '{query}', bypassing cache read")

    # retrieve papers
    print(f"\nRetrieving top {top_k} papers for query: '{query}'")
    papers = get_papers_with_abstracts(query, top_k=top_k)
    print(f"Retrieved {len(papers)} papers")

    paper_context = format_papers_for_prompt(papers)

    system_msg = (
        "You are a research summarization assistant. You synthesize information "
        "across retrieved papers to help researchers understand a topic. "
        "Critical rule: only state claims that are directly supported by the paper "
        "abstracts provided. Do not extrapolate or invent details about a paper's "
        "contributions. If a retrieved paper is not actually about the user's query, "
        "say so explicitly rather than forcing it into the narrative."
    )

    user_msg = (
        f"User query: \"{query}\"\n\n"
        f"Below are {len(papers)} papers retrieved from a research database via "
        f"semantic search. Some may directly match the query; others may be only "
        f"tangentially related - that is normal in semantic retrieval.\n\n"
        f"Write a summary following these rules strictly:\n\n"
        f"1. Open with substantive content. Do NOT begin with generic phrases like "
        f"\"This is a rapidly evolving field\" or \"has garnered significant attention.\"\n"
        f"2. State briefly which retrieved papers are directly relevant to the query. "
        f"If some are only tangentially related, name them and set them aside or "
        f"explain the loose connection - do not pretend they are central to the topic.\n"
        f"3. For the directly relevant papers, describe their actual methods and "
        f"contributions, citing each by its title. Do not invent contributions.\n"
        f"4. Identify real methodological or thematic connections between papers "
        f"where they exist. Do not invent connections.\n"
        f"5. Do NOT mention citation counts unless at least one paper has a notable "
        f"count (50+). If all are zero or low, ignore citations entirely.\n"
        f"6. Close with a substantive observation about the state of the area based "
        f"on what is in front of you. Do NOT use generic closers like \"vibrant and "
        f"dynamic field\" or \"crucial for advancing the field.\"\n"
        f"7. Length: 250-400 words. Tighter is better.\n"
        f"8. Plain prose only. No bullet points, numbered lists, headers, or "
        f"markdown formatting.\n\n"
        f"PAPERS:\n\n{paper_context}\n\n"
        f"Write the summary now."
    )

    print("Calling Groq API...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    summary = response.choices[0].message.content

    papers_used = papers[["arxiv_id", "title", "authors", "citation_count", "similarity"]]

    # write to cache
    if use_cache:
        cache = _load_cache()
        cache[key] = {
            "query": query,
            "summary": summary,
            "papers_used": papers_used.to_dict(orient="records"),
            "cached_at": datetime.now().isoformat(),
        }
        _save_cache(cache)
        print(f"Cached result under key {key[:8]}...")

    return {
        "query": query,
        "summary": summary,
        "papers_used": papers_used,
        "from_cache": False,
    }


if __name__ == "__main__":
    # quick test - run twice, the second call should hit cache
    test_query = "fairness and bias in machine learning"

    print("\n=== First call (should be cache MISS) ===")
    result1 = summarize_topic(test_query, top_k=8)
    print(f"\nFrom cache: {result1['from_cache']}")

    print("\n\n=== Second call (should be cache HIT) ===")
    result2 = summarize_topic(test_query, top_k=8)
    print(f"\nFrom cache: {result2['from_cache']}")
    print(f"Same summary text: {result1['summary'] == result2['summary']}")

    print("\n\n=== Cache stats ===")
    cache_stats()