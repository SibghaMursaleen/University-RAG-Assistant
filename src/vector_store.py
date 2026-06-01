import os
import time
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Free-tier limit is 100 embed requests/minute.
# We batch chunks and sleep between batches to stay under the limit.
_BATCH_SIZE = 20      # chunks per API batch call
_BATCH_DELAY = 15     # seconds to wait between batches (safe margin)

def create_vector_store(chunks: List[Document], embeddings: GoogleGenerativeAIEmbeddings, save_path: str) -> FAISS:
    """
    Creates a new FAISS vector database from text chunks, computes embeddings, and saves it locally.
    
    Theory:
    What is FAISS (Facebook AI Similarity Search)?
    - FAISS is a library developed by Meta for extremely fast, high-performance similarity search on vectors.
    - Instead of doing a brute-force search comparing a query vector to every document vector sequentially,
      FAISS indexes the vectors using advanced data structures (like flat index, HNSW, or IVF).
    - It allows retrieving the top-K nearest neighbors in logarithmic time O(log N).
    
    What happens during batched embedding?
    1. We split the chunks into small batches of _BATCH_SIZE.
    2. Each batch is sent to the Gemini Embedding API (one API call per chunk).
    3. After each batch, we sleep for _BATCH_DELAY seconds so we stay under the
       free-tier rate limit of 100 requests/minute.
    4. The resulting vectors are progressively merged into a single FAISS index.
    5. The final index is saved to disk so we never need to re-embed!
    
    Free Tier Rate Limit:
    - Google Gemini Embedding free tier: 100 requests / minute
    - Each chunk = 1 request, so for 100+ chunks we must pace our calls.
    
    Args:
        chunks (List[Document]): The list of split text chunks.
        embeddings (GoogleGenerativeAIEmbeddings): The initialized embedding client.
        save_path (str): The folder path where FAISS database files should be saved.
        
    Returns:
        FAISS: The constructed vector store object.
    """
    total = len(chunks)
    print(f"[Vector Store] Creating FAISS index for {total} chunks (batch size: {_BATCH_SIZE})...")

    vector_db = None
    for i in range(0, total, _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        total_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
        print(f"  -> Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_db is None:
            # First batch — create a new FAISS index
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            # Subsequent batches — build a temporary index and merge it in
            batch_db = FAISS.from_documents(batch, embeddings)
            vector_db.merge_from(batch_db)

        # Respect rate limit between batches (skip delay after last batch)
        if i + _BATCH_SIZE < total:
            print(f"  [Rate-limit pause] Wait {_BATCH_DELAY}s before next batch...")
            time.sleep(_BATCH_DELAY)

    # Save the merged index locally
    os.makedirs(save_path, exist_ok=True)
    vector_db.save_local(save_path)
    print(f"[Vector Store] FAISS database built and saved at '{save_path}'.")
    return vector_db


def load_vector_store(save_path: str, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Loads an existing local FAISS vector database from disk.
    
    Args:
        save_path (str): The folder path where the FAISS database is stored.
        embeddings (GoogleGenerativeAIEmbeddings): The embedding model used when creating the store.
        
    Returns:
        FAISS: The loaded vector store.
    """
    if not os.path.exists(save_path) or not os.path.exists(os.path.join(save_path, "index.faiss")):
        raise FileNotFoundError(f"No local FAISS database found at path: {save_path}")
        
    print(f"[Vector Store] Loading local FAISS database from '{save_path}'...")
    
    # Load the FAISS database and bind the embedding model.
    # The embedding model is required because FAISS must embed search queries too!
    vector_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    
    print("[Vector Store] FAISS database loaded successfully.")
    return vector_db
