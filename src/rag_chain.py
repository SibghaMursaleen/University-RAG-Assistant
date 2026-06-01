import os
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def get_gemini_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM client.
    
    Theory:
    Why use Gemini Flash?
    - `gemini-2.5-flash` is optimized for high-speed, cost-effective, and highly intelligent 
      text generation. It is extremely fast and handles large contexts perfectly.
      
    Why set Temperature to 0.2?
    - Temperature controls randomness.
    - 0.0 to 0.3 (Low): Makes the model highly deterministic, precise, and literal. Excellent 
      for factual tasks like QA on a syllabus or technical specs.
    - 0.7 to 1.0 (High): Makes the model creative and diverse. 
    - For RAG, we want low temperature (e.g., 0.2) to prevent the model from inventing 
      facts outside our retrieved syllabus text (preventing hallucinations).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or "your_gemini_api_key_here" in api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm

def create_rag_chain(retriever: VectorStoreRetriever, llm: ChatGoogleGenerativeAI):
    """
    Assembles the prompt template and binds the retriever and LLM into a LangChain Retrieval Chain.
    
    Theory:
    How does a Stuff Documents Chain work?
    - It takes a list of retrieved documents, extracts their text, joins ("stuffs") them together,
      and injects them into the prompt template variable `{context}`.
    - It then passes the user's question as `{input}`.
    
    Why write a strict System Prompt?
    - Large language models are highly conversational. Without guidelines, they might guess answers 
      or draw from their general training data when asked about your specific course policies.
    - Our strict system prompt forces the LLM to restrict its knowledge bounds specifically 
      to the retrieved course documentation.
      
    What is the return structure of create_retrieval_chain?
    - The chain takes a dictionary: `{"input": "your question"}`.
    - It returns a dictionary containing:
        - `input` (str): The original question.
        - `context` (List[Document]): The actual source chunks retrieved from FAISS.
        - `answer` (str): The final generated answer from the LLM.
    
    This is extremely beautiful because we can show the user BOTH the final answer 
    and the exact source documents that supported it!
    """
    
    # 1. Define the System Prompt
    system_prompt = (
        "You are an expert, helpful, and friendly University RAG Assistant.\n"
        "Your goal is to answer students' questions about course policies, syllabus details, and lectures "
        "using ONLY the retrieved academic context provided below.\n\n"
        "Rules:\n"
        "1. Restrict your answers strictly to the provided context. Do not make assumptions or extrapolate.\n"
        "2. If the answer cannot be found in the retrieved context, politely state: "
        "'I am sorry, but I couldn't find that information in the uploaded course documents. "
        "Please check with your instructor or office hours.'\n"
        "3. Provide clear, professional, and well-structured markdown answers (use lists, bold text, or tables if helpful).\n\n"
        "Retrieved Context:\n"
        "-----------------\n"
        "{context}\n"
        "-----------------\n"
    )
    
    # 2. Build the Chat Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # 3. Create the Stuff Documents Chain (which combines retrieved docs into the `{context}` variable)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Create the complete Retrieval Chain (connects retriever to QA chain)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("[RAG Chain] Successfully constructed the complete Retrieval-QA chain.")
    return rag_chain
