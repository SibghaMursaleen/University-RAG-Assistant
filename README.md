# 🎓 University RAG Assistant

An **AI-powered Retrieval-Augmented Generation (RAG)** assistant that lets students ask natural language questions about their course syllabi, lecture notes, and assignments — and receive precise, grounded answers with page-level citations.

Built as a **portfolio-quality AI project** demonstrating end-to-end LLM engineering: from raw PDF parsing to semantic vector search to a polished production UI.

---

## ✨ Features

- 📄 **Multi-PDF Ingestion**: Upload any course PDF (syllabus, lecture slides, notes) via the web UI
- 🧠 **Semantic Search with FAISS**: Fast, local vector similarity search using Gemini embeddings
- 🤖 **Gemini-Powered Answers**: Grounded, citation-backed answers using `gemini-2.5-flash`
- 📌 **Source Citations**: Every answer shows the exact PDF filename and page number used
- 🛡️ **Hallucination Guard**: The assistant politely refuses to answer out-of-scope questions
- 💬 **Persistent Chat History**: Full session-level conversation memory in the UI
- 🎨 **Premium Dark UI**: Glassmorphism design with animated chat bubbles and typing indicators

---

## 🏗️ Architecture

```
PDF Documents
      ↓
 PyPDF Loader  (src/document_loader.py)
      ↓
Text Splitter  (src/text_splitter.py)   ← RecursiveCharacterTextSplitter, chunk=500, overlap=100
      ↓
Gemini Embeddings  (src/embeddings.py)  ← models/gemini-embedding-001 → 768-dim vectors
      ↓
   FAISS Index  (src/vector_store.py)   ← Saved locally in vector_db/
      ↓
──────────────────────────────────
      ↓
 User Question → Embed → MMR Search  (src/retriever.py)   k=3 diverse chunks
      ↓
Prompt Assembly + Gemini LLM  (src/rag_chain.py)
      ↓
   Final Answer + Source Citations
      ↓
 Streamlit Web UI  (app/streamlit_app.py)
```

---

## 🗂️ Project Structure

```
university-rag-assistant/
├── app/
│   └── streamlit_app.py        # Premium Streamlit chat UI
├── data/
│   └── pdfs/                   # Place your PDF documents here
├── src/
│   ├── document_loader.py      # PDF parsing with metadata
│   ├── text_splitter.py        # Recursive chunking with overlap
│   ├── embeddings.py           # Gemini embedding client
│   ├── vector_store.py         # FAISS index creation & persistence
│   ├── retriever.py            # MMR similarity search
│   └── rag_chain.py            # LangChain retrieval chain + LLM
├── vector_db/                  # Auto-generated FAISS index files
├── .env                        # Your API key (never commit this!)
├── .env.template               # Template for new contributors
├── .gitignore
├── requirements.txt
├── main.py                     # CLI interactive session
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd university-rag-assistant

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the template and add your key
copy .env.template .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=AIzaSy...your_key_here
```

Get a free API key at 👉 https://aistudio.google.com/

### 3. Launch the Web App

```bash
.\.venv\Scripts\streamlit run app\streamlit_app.py
```

Open your browser at **http://localhost:8501**

### 4. Add Your Documents

1. Click **Upload Course Documents** in the sidebar
2. Drop your syllabus or lecture PDFs
3. Click **⚡ Build Knowledge Base**
4. Start chatting!

### 5. (Optional) CLI Mode

```bash
.\.venv\Scripts\python main.py
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Framework | LangChain / LangChain Classic |
| Generator | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 (768-dim) |
| Vector DB | FAISS (local, CPU) |
| PDF Parsing | PyPDF + PyMuPDF |
| Web UI | Streamlit |
| Config | python-dotenv |

---

## 🧠 What You Learn Building This

- How **embeddings** convert text to 768-dimensional geometric vectors
- How **FAISS** indexes vectors for sub-millisecond similarity search
- How **RAG** retrieves relevant context before generating answers
- How **LangChain** chains retrieval → prompt → LLM in a clean pipeline
- How to prevent **hallucinations** with strict system prompts
- How to build **production-quality AI UIs** with Streamlit

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
