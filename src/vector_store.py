import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(chunks: List[Document], embeddings: GoogleGenerativeAIEmbeddings, save_path: str) -> FAISS:
    """
    Creates a new FAISS vector database from text chunks, computes embeddings, and saves it locally.
    
    Theory:
    What is FAISS (Facebook AI Similarity Search)?
    - FAISS is a library developed by Meta for extremely fast, high-performance similarity search on vectors.
    - Instead of doing a brute-force search comparing a query vector to every document vector sequentially,
      FAISS indexes the vectors using advanced data structures (like flat index, HNSW, or IVF).
    - It allows retrieving the top-K nearest neighbors in logarithmic time O(log N).
    
    What happens during FAISS.from_documents?
    1. For every chunk in the list:
        - The text content is sent to the Gemini Embedding API.
        - The API returns a 768-dimensional float vector.
    2. FAISS stores both the raw text content, the metadata (page number, source path), and the generated 
       768-dimensional vector inside a fast in-memory index.
    3. The entire database (index and text mapping) is saved to the disk at `save_path` (contains index.faiss 
       and index.pkl) so we don't have to re-compute embeddings every time!
    
    Args:
        chunks (List[Document]): The list of split text chunks.
        embeddings (GoogleGenerativeAIEmbeddings): The initialized embedding client.
        save_path (str): The folder path where FAISS database files should be saved.
        
    Returns:
        FAISS: The constructed vector store object.
    """
    print(f"[Vector Store] Creating new FAISS index with {len(chunks)} chunks...")
    
    # Compute embeddings and create the FAISS index
    # Note: This will trigger API calls to embed all the text chunks.
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally to avoid repeated embedding costs
    os.makedirs(save_path, exist_ok=True)
    vector_db.save_local(save_path)
    print(f"[Vector Store] Successfully built and saved FAISS database locally at '{save_path}'.")
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
    
    # Load the FAISS database and bind the embedding model
    # Note: The embedding model is required because FAISS needs to compute embeddings for any search queries!
    vector_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    
    print("[Vector Store] FAISS database loaded successfully.")
    return vector_db
