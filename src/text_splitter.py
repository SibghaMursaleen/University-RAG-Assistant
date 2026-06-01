from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of page-level Document objects into smaller, overlapping chunks.
    
    Theory:
    Why do we need Chunking?
    - If we feed entire pages or documents into a vector database, the embeddings will represent a mix of 
      many different topics, diluting the search accuracy.
    - Large text segments also exceed the context window limitations of LLMs or cost too much.
    - Splitting documents into small, focused chunks (e.g., paragraphs) ensures that the search retrieves
      highly specific sections directly answering the user's query.
      
    Why use RecursiveCharacterTextSplitter?
    - Simple splitting by fixed length (e.g., every 500 characters) cuts text in the middle of words or sentences.
    - Recursive splitting uses a list of separators: `["\\n\\n", "\\n", " ", ""]`.
    - It tries to keep paragraphs together first (`\\n\\n`), then sentences (`\\n`), then words (` `) to preserve 
      semantic integrity and readability.
      
    Why do we need Chunk Overlap?
    - Consider a fact split exactly at the chunk boundary: Part A is in Chunk 1, Part B is in Chunk 2.
    - The meaning is lost! 
    - Overlap ensures that a sliding window carries over some characters from the previous chunk. 
      This keeps sentences intact and maintains critical surrounding context.
      
    Args:
        documents (List[Document]): The raw page-level documents extracted from the loader.
        chunk_size (int): The target maximum size of each chunk (in characters).
        chunk_overlap (int): The overlap size between consecutive chunks (in characters).
        
    Returns:
        List[Document]: A list of chunked Document objects with updated content and inherited metadata.
    """
    # Initialize the recursive character splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    # Split the documents
    chunks = splitter.split_documents(documents)
    
    print(f"[Text Splitter] Split {len(documents)} source pages into {len(chunks)} text chunks.")
    return chunks
