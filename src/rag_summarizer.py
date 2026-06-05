"""
RAG-based topic summarization using Groq's LLM API.

Workflow:
1. User provides a query (e.g., "graph neural networks for drug discovery")
2. FAISS retrieves the top-N most semantically relevant papers
3. Paper metadata + abstracts are formatted into a structured prompt
4. Groq's LLM generates a coherent summary of the research area
"""

import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from search_engine import search_papers

# load .env from project root (one level up from src/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

print(f"DEBUG: GROQ_API_KEY loaded: {GROQ_API_KEY[:10]}...")

# initialize Groq client once at import time
client = Groq(api_key=GROQ_API_KEY)

# model choice — llama 3.3 70b is the best general-purpose model on Groq's free tier
# if this errors with "model not found", check https://console.groq.com/docs/models
# for the current list and replace the name
MODEL = "llama-3.3-70b-versatile"


def get_papers_with_abstracts(query, top_k=10,
                              papers_path="../models/papers_with_topics.csv"):
    """
    Retrieves top-k papers for a query via FAISS, then joins with the full
    papers CSV to include abstracts (which search_papers does not return).
    Returns a DataFrame sorted by similarity.
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
    """
    Formats papers into a structured context block for the LLM prompt.
    Truncates long fields so the prompt stays within reasonable token limits.
    """
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


def summarize_topic(query, top_k=8, temperature=0.3, max_tokens=1500):
    """
    Generates a RAG-based summary of a research topic.

    Args:
        query: search string (e.g., "transformers for time series")
        top_k: number of papers to retrieve and include in the prompt
        temperature: LLM sampling temperature (lower = more focused)
        max_tokens: max tokens in the generated summary

    Returns:
        dict with keys:
            'query'       : the input query
            'summary'     : LLM-generated summary text
            'papers_used' : DataFrame of the papers used in context
    """
    print(f"\nRetrieving top {top_k} papers for query: '{query}'")
    papers = get_papers_with_abstracts(query, top_k=top_k)
    print(f"Retrieved {len(papers)} papers")

    paper_context = format_papers_for_prompt(papers)

    system_msg = (
        "You are a research summarization assistant. You help early-career "
        "researchers quickly understand emerging areas of research by "
        "synthesizing information across multiple related papers."
    )

    user_msg = (
        f"The user is researching the area: \"{query}\"\n\n"
        f"Below are the {len(papers)} most relevant papers retrieved from a "
        f"research database. For each, you have its title, authors, citation "
        f"count, and abstract.\n\n"
        f"Your task: Write a clear, well-structured summary (300-500 words) "
        f"of this research area based on these papers. The summary should:\n"
        f"1. Identify the main themes, approaches, and methods across the papers\n"
        f"2. Highlight notable findings, techniques, or open challenges\n"
        f"3. Note which papers appear most influential based on citation count\n"
        f"4. Be written for an early-career researcher entering this field\n"
        f"5. Use plain prose. No bullet points or numbered lists.\n\n"
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

    return {
        "query": query,
        "summary": summary,
        "papers_used": papers[["arxiv_id", "title", "authors", "citation_count", "similarity"]]
    }


if __name__ == "__main__":
    test_query = "graph neural networks for molecular property prediction"

    result = summarize_topic(test_query, top_k=8)

    print("\n" + "=" * 60)
    print(f"QUERY: {result['query']}")
    print("=" * 60)

    print("\nPAPERS USED:")
    for i, row in result["papers_used"].iterrows():
        print(f"  - {row['title']} (citations: {row['citation_count']})")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(result["summary"])