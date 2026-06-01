from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(vector_db: FAISS, search_type: str = "similarity", k: int = 3) -> VectorStoreRetriever:
    """
    Converts a FAISS vector database into a LangChain Retriever object.
    
    Theory:
    What is a Retriever?
    - A vector database stores, indexes, and queries vector points.
    - A Retriever is a LangChain interface that wraps any retrieval mechanism (e.g., vector database, 
      hybrid search, or search engines) and exposes a single simple function: `get_relevant_documents(query)`.
      
    What is the difference between search types?
    - "similarity" (Standard Cosine Similarity):
        - Simply fetches the top-K chunks that are closest to the query in embedding space.
        - Fast and effective, but can sometimes return highly redundant information (e.g., 3 chunks 
          expressing the same sentence slightly differently).
    - "mmr" (Maximal Marginal Relevance):
        - Solves the redundancy problem!
        - First retrieves a larger set of matching chunks, and then re-ranks and filters them to optimize 
          two things: similarity to the query, and *diversity* among themselves.
        - Perfect for when you want a broad, comprehensive answer from multiple parts of a syllabus!
        
    Args:
        vector_db (FAISS): The loaded FAISS database.
        search_type (str): The search algorithm to use ("similarity" or "mmr").
        k (int): The number of relevant documents/chunks to retrieve.
        
    Returns:
        VectorStoreRetriever: The configured LangChain retriever.
    """
    print(f"[Retriever] Initializing retriever (Type: {search_type}, k: {k})...")
    
    # Check that search type is valid
    if search_type not in ["similarity", "mmr"]:
        print(f"[!] Warning: Invalid search_type '{search_type}'. Defaulting to 'similarity'.")
        search_type = "similarity"
        
    # Build the retriever
    retriever = vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    return retriever
