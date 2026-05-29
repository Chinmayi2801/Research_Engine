import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer


# load model once
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def build_faiss_index(embeddings_path="../models/paper_embeddings.npy",
                     save_path="../models/faiss_index.bin"):
    """
    Builds a FAISS index from saved embeddings and saves it to disk.
    Returns the index.
    """
    print(f"\nLoading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path).astype("float32")
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # normalize embeddings for cosine similarity
    # FAISS by default uses L2 distance, but with normalized vectors L2 ranks the same as cosine
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    
    # IndexFlatIP — inner product. With normalized vectors, this equals cosine similarity.
    # Flat means exact search (no approximation). Fine for our dataset size.
    index = faiss.IndexFlatIP(dimension)
    
    # add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    
    # save index
    os.makedirs("../models", exist_ok=True)
    faiss.write_index(index, save_path)
    print(f"Index saved to {save_path}")
    
    return index


def load_faiss_index(index_path="../models/faiss_index.bin"):
    """
    Loads a previously built FAISS index from disk.
    """
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index


def search_papers(query, top_k=10,
                  index_path="../models/faiss_index.bin",
                  papers_path="../models/papers_with_topics.csv"):
    """
    Takes a query string, returns top-k most similar papers as a dataframe.
    """
    index = load_faiss_index(index_path)
    df = pd.read_csv(papers_path)
    
    # embed the query
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # search
    similarities, indices = index.search(query_embedding, top_k)
    
    # build results dataframe
    results = df.iloc[indices[0]].copy()
    results["similarity"] = similarities[0]
    
    return results[["arxiv_id", "title", "authors", "published_date",
                    "citation_count", "topic", "similarity"]]


def find_similar_papers(paper_idx, top_k=10,
                       index_path="../models/faiss_index.bin",
                       embeddings_path="../models/paper_embeddings.npy",
                       papers_path="../models/papers_with_topics.csv"):
    """
    Given an index of a paper in the dataset, returns the top-k most similar papers.
    Used for the 'related papers' feature.
    """
    index = load_faiss_index(index_path)
    embeddings = np.load(embeddings_path).astype("float32")
    df = pd.read_csv(papers_path)
    
    # get this paper's embedding (already normalized in index, normalize here too)
    paper_embedding = embeddings[paper_idx:paper_idx+1].copy()
    faiss.normalize_L2(paper_embedding)
    
    # search top_k+1 because the paper itself will be the first result
    similarities, indices = index.search(paper_embedding, top_k + 1)
    
    # drop the paper itself (first result)
    indices = indices[0][1:]
    similarities = similarities[0][1:]
    
    results = df.iloc[indices].copy()
    results["similarity"] = similarities
    
    return results[["arxiv_id", "title", "authors", "published_date",
                    "citation_count", "topic", "similarity"]]


if __name__ == "__main__":
    # build the index
    index = build_faiss_index()
    
    # test 1: keyword search
    print("\n" + "="*60)
    print("TEST 1: Search query = 'reinforcement learning for robotics'")
    print("="*60)
    results = search_papers("reinforcement learning for robotics", top_k=5)
    for i, row in results.iterrows():
        print(f"\n  {row['title']}")
        print(f"  Similarity: {row['similarity']:.3f} | Topic: {row['topic']}")
    
    # test 2: related papers
    print("\n" + "="*60)
    print("TEST 2: Find papers related to paper at index 0")
    print("="*60)
    df = pd.read_csv("../models/papers_with_topics.csv")
    print(f"\nReference paper: {df['title'].iloc[0]}")
    print(f"\nMost similar papers:")
    
    similar = find_similar_papers(0, top_k=5)
    for i, row in similar.iterrows():
        print(f"\n  {row['title']}")
        print(f"  Similarity: {row['similarity']:.3f} | Topic: {row['topic']}")