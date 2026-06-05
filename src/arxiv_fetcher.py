import requests
import pandas as pd
import time
import os
import xml.etree.ElementTree as ET

def fetch_arxiv_papers(topic, max_results=500):
    """
    Fetches papers from arXiv API for a given topic.
    Returns a cleaned pandas DataFrame.
    """
    
    print(f"Fetching papers for topic: {topic}")
    
    base_url = "http://export.arxiv.org/api/query"
    
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    
    root = ET.fromstring(response.content)
    
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    papers = []
    
    entries = root.findall("atom:entry", namespace)
    print(f"Found {len(entries)} papers")
    
    for entry in entries:
        try:
            arxiv_id = entry.find("atom:id", namespace).text
            arxiv_id = arxiv_id.split("/abs/")[-1]
            
            title = entry.find("atom:title", namespace).text
            title = title.replace("\n", " ").strip()
            
            abstract_text = entry.find("atom:summary", namespace).text
            abstract_text = abstract_text.replace("\n", " ").strip()
            
            published = entry.find("atom:published", namespace).text
            published = published[:10]
            
            authors = []
            for author in entry.findall("atom:author", namespace):
                name = author.find("atom:name", namespace).text
                authors.append(name)
            authors_str = ", ".join(authors)
            
            categories = []
            for cat in entry.findall("atom:category", namespace):
                categories.append(cat.get("term"))
            categories_str = ", ".join(categories)
            
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract_text,
                "authors": authors_str,
                "published_date": published,
                "categories": categories_str,
                "topic_query": topic
            })
            
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue
    
    df = pd.DataFrame(papers)
    return df


def save_papers(df, topic):
    """
    Saves the fetched papers to the data folder as CSV.
    """
    os.makedirs("../data", exist_ok=True)
    
    clean_topic = topic.replace(" ", "_").lower()
    filepath = f"../data/{clean_topic}_papers.csv"
    
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} papers to {filepath}")
    return filepath

def fetch_arxiv_papers_by_date_range(topic, start_date, end_date, max_results=200):
    """
    Fetches papers from arXiv for a topic within a specific date range.
    Used to fetch older papers with citation history.
    """
    print(f"Fetching papers for topic: {topic} between {start_date} and {end_date}")
    
    base_url = "http://export.arxiv.org/api/query"
    
    # arxiv date format in queries: YYYYMMDDHHMM
    start_fmt = start_date.replace("-", "") + "0000"
    end_fmt = end_date.replace("-", "") + "2359"
    
    params = {
        "search_query": f"all:{topic} AND submittedDate:[{start_fmt} TO {end_fmt}]",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    
    root = ET.fromstring(response.content)
    
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    papers = []
    entries = root.findall("atom:entry", namespace)
    print(f"Found {len(entries)} papers in date range")
    
    for entry in entries:
        try:
            arxiv_id = entry.find("atom:id", namespace).text
            arxiv_id = arxiv_id.split("/abs/")[-1]
            
            title = entry.find("atom:title", namespace).text.replace("\n", " ").strip()
            abstract_text = entry.find("atom:summary", namespace).text.replace("\n", " ").strip()
            published = entry.find("atom:published", namespace).text[:10]
            
            authors = []
            for author in entry.findall("atom:author", namespace):
                name = author.find("atom:name", namespace).text
                authors.append(name)
            authors_str = ", ".join(authors)
            
            categories = []
            for cat in entry.findall("atom:category", namespace):
                categories.append(cat.get("term"))
            categories_str = ", ".join(categories)
            
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract_text,
                "authors": authors_str,
                "published_date": published,
                "categories": categories_str,
                "topic_query": topic
            })
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue
    
    df = pd.DataFrame(papers)
    return df


if __name__ == "__main__":
    topic = "machine learning"
    
    df = fetch_arxiv_papers(topic, max_results=100)
    
    if df is not None and len(df) > 0:
        print(df.head())
        print(f"\nTotal papers fetched: {len(df)}")
        print(f"\nColumns: {df.columns.tolist()}")
        save_papers(df, topic)
    else:
        print("No papers fetched, something went wrong.")