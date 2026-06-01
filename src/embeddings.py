import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini Embedding Model.
    
    Theory:
    What is an Embedding Model?
    - Deep learning models understand text as mathematical vectors (lists of floats) in high-dimensional space.
    - An embedding model maps semantic meaning into geometric space.
    - Google's `models/embedding-001` maps text chunks into a 768-dimensional vector!
    - When we embed text:
        - "CS 101 syllabus" and "Computer Science Course Info" will map to vectors very close in space.
        - "Baking apple pie" will map to a vector far away.
    - The closeness of vectors is measured using Cosine Similarity or L2 distance.
    
    How does this package authenticate?
    - The `GoogleGenerativeAIEmbeddings` class automatically reads the `GOOGLE_API_KEY` variable 
      from the environment.
    - We load the key using `dotenv` to keep your credentials safe and out of git!
    
    Returns:
        GoogleGenerativeAIEmbeddings: The embedding client.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or "your_gemini_api_key_here" in api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please add your actual Google Gemini API key to the '.env' file."
        )
    
    # Initialize the embeddings wrapper using Gemini's gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings
