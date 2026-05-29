import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os


def fit_topic_model(embeddings_path="../models/paper_embeddings.npy", 
                    papers_path="../models/paper_embeddings_papers.csv",
                    save_path="../models/bertopic_model"):
    """
    Fits BERTopic on existing embeddings and saves the model + topic assignments.
    """
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"Loading papers from {papers_path}")
    df = pd.read_csv(papers_path)
    
    print(f"\nFitting BERTopic on {len(df)} papers...")
    
    # use the same sentence transformer for consistency
    # we don't pass our embeddings directly — we let BERTopic compute internally
    # because BERTopic also needs raw text for c-TF-IDF labeling
    texts = df["text_for_embedding"].fillna("").tolist()
    
    # initialize BERTopic with defaults
    # min_topic_size lowered since we have a small dataset
    vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))    
    
    topic_model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        vectorizer_model=vectorizer,
        min_topic_size=5,
        verbose=True
    )
    
    # fit and get topic assignments
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    # add topic assignments to dataframe
    df["topic"] = topics
    
    print(f"\nFound {len(set(topics))} unique topics (including -1 for outliers)")
    
    # show topic info
    print("\n--- Topic Info ---")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(20))
    
    # show top words per topic
    print("\n--- Top words per topic ---")
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            word_list = [word for word, _ in words[:8]]
            count = sum(1 for t in topics if t == topic_id)
            print(f"Topic {topic_id} ({count} papers): {', '.join(word_list)}")
    
    # save the model
    os.makedirs("../models", exist_ok=True)
    topic_model.save(save_path, serialization="safetensors")
    print(f"\nModel saved to {save_path}")
    
    # save the dataframe with topic assignments
    df.to_csv("../models/papers_with_topics.csv", index=False)
    print(f"Topic assignments saved to ../models/papers_with_topics.csv")
    
    return topic_model, df


def load_topic_model(model_path="../models/bertopic_model"):
    """
    Loads a previously saved BERTopic model.
    """
    print(f"Loading topic model from {model_path}")
    topic_model = BERTopic.load(model_path)
    print("Loaded.")
    return topic_model


if __name__ == "__main__":
    topic_model, df = fit_topic_model()
    
    print("\n--- Sanity check ---")
    print(f"Total papers: {len(df)}")
    print(f"Papers per topic:")
    print(df["topic"].value_counts())