<h1 align="center">🎓 University RAG Assistant</h1>

<p align="center">
  An AI-powered academic assistant that answers student questions using your own course documents.<br/>
  Upload any syllabus or lecture PDF and get precise, source-cited answers — no hallucinations.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-00599C?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 📌 Overview

**University RAG Assistant** is a **Retrieval-Augmented Generation (RAG)** pipeline built for students and educators. It lets you upload course PDFs — syllabi, lecture notes, assignment briefs — and then ask natural language questions about them. The assistant retrieves the most relevant document chunks using **FAISS semantic search**, injects them into a strict prompt, and uses **Gemini 2.5 Flash** to generate grounded, accurate answers with full source citations.

> **Key:** The system never fabricates information. If the answer isn't in your documents, it says so honestly.

---

## ⚙️ How It Works

| Step | Stage | Description |
|------|-------|-------------|
| 1 | **Document Loading** | PDFs are parsed page-by-page using `PyPDFLoader`, preserving metadata (source file, page number) |
| 2 | **Text Splitting** | Pages are chunked into 500-token segments with 100-token overlap using `RecursiveCharacterTextSplitter` |
| 3 | **Embedding** | Each chunk is embedded into a 768-dimensional vector using Google's `gemini-embedding-001` model |
| 4 | **Indexing** | Vectors are stored in a local **FAISS** index with batched API calls to respect free-tier rate limits |
| 5 | **Retrieval** | At query time, **MMR (Maximal Marginal Relevance)** retrieves the top-3 diverse, relevant chunks |
| 6 | **Generation** | A strict system prompt + retrieved context is sent to **Gemini 2.5 Flash** (temp=0.2) for grounded answers |
| 7 | **Citation** | The UI displays the exact source file and page number for every answer |

---

## 📁 Project Structure

```
University-RAG-Assistant/
│
├── app/
│   └── streamlit_app.py     # Full Streamlit UI (dark glassmorphism theme)
│
├── src/
│   ├── document_loader.py   # PDF parsing with PyPDFLoader
│   ├── text_splitter.py     # Recursive chunking with overlap
│   ├── embeddings.py        # Google Gemini embedding model wrapper
│   ├── vector_store.py      # Batched FAISS index creation & loading
│   ├── retriever.py         # MMR retriever configuration
│   └── rag_chain.py         # LangChain retrieval chain assembly
│
├── data/
│   └── pdfs/                # Place your course PDFs here (gitignored)
│
├── vector_db/
│   └── faiss_index/         # Auto-generated FAISS index (gitignored)
│
├── main.py                  # CLI interactive session entry point
├── requirements.txt         # Python dependencies
├── .env.template            # API key configuration template
└── .gitignore
```

> **Note:** `data/pdfs/` and `vector_db/` are excluded from Git. The FAISS index is regenerated automatically when you upload PDFs and click **Build Knowledge Base**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com/)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/SibghaMursaleen/University-RAG-Assistant.git
cd University-RAG-Assistant
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Your API Key

```bash
# Copy the template
cp .env.template .env
```

Open `.env` and replace the placeholder with your actual key:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### Step 5 — Run the App

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

| Action | How |
|--------|-----|
| Upload PDFs | Use the sidebar file uploader |
| Build index | Click **⚡ Build Knowledge Base** |
| Ask questions | Type in the chat input at the bottom |
| Clear documents | Click **🗑️ Clear Knowledge Base** in the sidebar |

---

## 🎨 Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_size` | `500` | Token size of each document chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |
| `retriever k` | `3` | Number of chunks retrieved per query |
| `search_type` | `mmr` | MMR for diverse retrieval (vs. similarity) |
| `temperature` | `0.2` | Low temp for factual, deterministic answers |
| `model` | `gemini-2.5-flash` | Gemini model used for generation |
| `embedding_model` | `gemini-embedding-001` | Model used for chunk + query embedding |

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10+** | Core language |
| **Streamlit 1.58** | Interactive web UI with dark glassmorphism theme |
| **LangChain** | RAG pipeline orchestration (retrieval chain, prompt templating) |
| **FAISS (CPU)** | High-performance local vector similarity search |
| **Google Gemini 2.5 Flash** | LLM for answer generation |
| **Google Gemini Embedding-001** | Text embedding into 768-dim vectors |
| **PyPDF / PyMuPDF** | PDF parsing and page extraction |
| **python-dotenv** | Secure environment variable management |

---

## ⚠️ Tips & Best Practices

- **PDF quality matters** — Scanned image PDFs won't parse well. Use text-based PDFs for best results.
- **Chunk your uploads** — The free Gemini embedding tier allows 100 requests/minute. The app automatically batches and rate-limits.
- **Keep temperature low** — The default `0.2` is ideal for factual RAG. Increasing it may cause the model to drift from your documents.
- **FAISS index is reusable** — Once built, the index is saved locally. You won't need to re-embed on every run.
- **Never commit your `.env`** — Your `GOOGLE_API_KEY` is in `.gitignore`. Keep it that way.

---

## 📄 License

This project is released under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<p align="center">
  Built with 🦜 LangChain &nbsp;·&nbsp; 🔍 FAISS &nbsp;·&nbsp; ✨ Gemini &nbsp;·&nbsp; 🎈 Streamlit
  <br/>
  <sub>Made by <a href="https://github.com/SibghaMursaleen">Sibgha Mursaleen</a></sub>
</p>
