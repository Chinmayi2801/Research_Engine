"""
RAG-based topic summarization using Groq's LLM API.

Workflow:
1. User provides a query
2. FAISS retrieves the top-N most semantically relevant papers
3. Paper metadata + abstracts are formatted into a structured prompt
4. Groq's LLM generates a coherent summary of the research area

Prompt tuned (Day 11) to address four issues observed in v1:
- Force-fitting tangentially-related retrieved papers into the query's narrative
- Misrepresenting paper contributions to fit the query
- Wasted sentences about citation counts when all are zero
- Generic boilerplate opening/closing paragraphs
"""

import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from search_engine import search_papers

# load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

print(f"DEBUG: GROQ_API_KEY loaded: {GROQ_API_KEY[:10]}...")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"


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
    """
    Formats papers into a structured context block for the LLM prompt.
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


def summarize_topic(query, top_k=8, temperature=0.3, max_tokens=800):
    """
    Generates a RAG-based summary of a research topic.
    """
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

    return {
        "query": query,
        "summary": summary,
        "papers_used": papers[["arxiv_id", "title", "authors", "citation_count", "similarity"]]
    }


if __name__ == "__main__":
    test_query = "quantum cryptography"

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