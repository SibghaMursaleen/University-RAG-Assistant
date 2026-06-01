import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF document and extracts its pages into a list of LangChain Document objects.
    
    Theory:
    Why do we need a Document Loader?
    - Raw PDFs are binary files containing layout instructions, fonts, and graphics, not plain text.
    - A PDF loader parses these binary tables and extracts the raw textual content, page by page.
    - Each page is loaded as a `Document` object in LangChain.
    - A `Document` contains two main attributes:
        1. `page_content` (str): The actual raw text extracted from that page.
        2. `metadata` (dict): Contextual metadata about the text (e.g., `source` file path, `page` number).
    
    Metadata is extremely crucial for a professional RAG assistant! It allows us to
    cite the exact page and document source when providing answers to users.
    
    Args:
        file_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        List[Document]: A list of page-by-page Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    # Initialize LangChain's PyPDFLoader
    loader = PyPDFLoader(file_path)
    
    # Load and parse the PDF page by page
    documents = loader.load()
    
    print(f"[Document Loader] Successfully loaded '{os.path.basename(file_path)}'. Extracted {len(documents)} pages.")
    return documents
