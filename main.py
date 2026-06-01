import os
import sys
from dotenv import load_dotenv

# Import our modular blocks
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, load_vector_store
from src.retriever import get_retriever
from src.rag_chain import get_gemini_llm, create_rag_chain

def initialize_system(pdf_dir: str, db_path: str):
    """
    Initializes the embedding model, loads/indexes documents if necessary,
    and returns a complete, runnable RAG chain.
    """
    # 1. Load Embeddings
    embeddings = get_embedding_model()
    
    # 2. Check if the vector DB index already exists
    if os.path.exists(os.path.join(db_path, "index.faiss")):
        print("[System] Local FAISS index detected. Skipping indexing phase.")
        vector_db = load_vector_store(db_path, embeddings)
    else:
        print("[System] No local index found. Proceeding with document parsing and indexing...")
        # Load and parse PDFs in our data folder
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF documents found in '{pdf_dir}'. "
                "Please place a course syllabus PDF inside this folder first!"
            )
            
        all_chunks = []
        for pdf_file in pdf_files:
            full_path = os.path.join(pdf_dir, pdf_file)
            docs = load_pdf(full_path)
            chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
            all_chunks.extend(chunks)
            
        # Create and save the vector DB
        vector_db = create_vector_store(all_chunks, embeddings, db_path)
        
    # 3. Setup Retriever (We use MMR for diverse retrieved context!)
    retriever = get_retriever(vector_db, search_type="mmr", k=3)
    
    # 4. Initialize LLM (We use the user's active model for best results)
    llm = get_gemini_llm(model_name="gemini-2.5-flash", temperature=0.2)
    
    # 5. Build RAG Chain
    rag_chain = create_rag_chain(retriever, llm)
    
    return rag_chain

def run_cli_session():
    # Enforce UTF-8 stdout for Windows consoles
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Load env keys
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or "your_gemini_api_key_here" in api_key:
        print("\n[!] ERROR: GOOGLE_API_KEY is not configured in the '.env' file.")
        print("Please add your key in 'u:\\Projects\\University-RAG-Assistant\\.env' before running.")
        sys.exit(1)
        
    pdf_dir = r"u:\Projects\University-RAG-Assistant\data\pdfs"
    db_path = r"u:\Projects\University-RAG-Assistant\vector_db\faiss_index"
    
    print("=" * 60)
    print("      🎓 UNIVERSITY RAG ASSISTANT - CLI INTERACTIVE 🎓      ")
    print("=" * 60)
    
    try:
        # Initialize RAG Pipeline
        rag_chain = initialize_system(pdf_dir, db_path)
        print("\n[System] System initialization complete. Ready for questions!")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            query = input("\nStudent Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Ending assistant session. Goodbye!")
                break
                
            print("[RAG Pipeline] Processing query and fetching sources...")
            
            # Invoke the retrieval chain
            # The chain automatically handles: query embedding -> FAISS retrieval -> prompt stuffing -> LLM response
            response = rag_chain.invoke({"input": query})
            
            print("\n" + "=" * 30 + " ANSWER " + "=" * 30)
            print(response["answer"])
            print("=" * 68)
            
            # Display source citations
            print("\n📚 Sources Cited:")
            retrieved_docs = response.get("context", [])
            if not retrieved_docs:
                print("  No source documents were retrieved.")
            else:
                for idx, doc in enumerate(retrieved_docs):
                    filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0) + 1
                    print(f"  [{idx+1}] File: {filename} (Page {page})")
                    # Optional: print snippet
                    snippet = doc.page_content.replace("\n", " ")[:120]
                    print(f"      Snippet: \"{snippet}...\"")
            print("=" * 68)
            
    except Exception as e:
        print(f"\n[!] Critical System Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cli_session()
