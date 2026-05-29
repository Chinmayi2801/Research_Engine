import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# load the embedding model once at import time
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def embed_text(text):
    """
    Takes a single text string and returns its 384-dim embedding as numpy array.
    """
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(384)
    
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def embed_papers(csv_path, save_path=None):
    """
    Reads the master CSV, embeds all paper abstracts, saves embeddings as .npy file.
    Returns the embeddings array and the dataframe.
    """
    print(f"\nLoading papers from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers")
    
    # combine title and abstract for richer embeddings
    df["text_for_embedding"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    
    texts = df["text_for_embedding"].tolist()
    
    print(f"\nGenerating embeddings for {len(texts)} papers...")
    print("This may take 1-2 minutes...")
    
    # batch encode is faster than one at a time
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # save embeddings
    if save_path is None:
        os.makedirs("../models", exist_ok=True)
        save_path = "../models/paper_embeddings.npy"
    
    np.save(save_path, embeddings)
    print(f"Embeddings saved to {save_path}")
    
    # also save the corresponding dataframe so we know which embedding maps to which paper
    df_save_path = save_path.replace(".npy", "_papers.csv")
    df.to_csv(df_save_path, index=False)
    print(f"Paper metadata saved to {df_save_path}")
    
    return embeddings, df


def load_embeddings(embeddings_path="../models/paper_embeddings.npy", papers_path=None):
    """
    Loads previously saved embeddings and the corresponding papers dataframe.
    """
    embeddings = np.load(embeddings_path)
    
    if papers_path is None:
        papers_path = embeddings_path.replace(".npy", "_papers.csv")
    
    df = pd.read_csv(papers_path)
    
    print(f"Loaded {embeddings.shape[0]} embeddings, dimension {embeddings.shape[1]}")
    print(f"Loaded {len(df)} papers")
    
    return embeddings, df


if __name__ == "__main__":
    csv_path = "../data/master_papers.csv"
    
    embeddings, df = embed_papers(csv_path)
    
    print("\n--- Sanity check ---")
    print(f"First paper title: {df['title'].iloc[0]}")
    print(f"First embedding shape: {embeddings[0].shape}")
    print(f"First 5 values of first embedding: {embeddings[0][:5]}")