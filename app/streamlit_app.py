import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.append(r"u:\Projects\University-RAG-Assistant")
load_dotenv()

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="University RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports of our pipeline modules ─────────────────────────────────────
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, load_vector_store
from src.retriever import get_retriever
from src.rag_chain import get_gemini_llm, create_rag_chain

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  ──  Premium dark glassmorphism design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #0d0f1a !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #818cf8 !important;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    margin-bottom: 1.2rem;
}
.sidebar-logo .logo-icon { font-size: 3rem; line-height: 1; }
.sidebar-logo .logo-title {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.4rem;
}
.sidebar-logo .logo-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Status badges ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.status-ok   { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.status-warn { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.status-err  { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.06) !important;
    border: 1.5px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chat messages container ── */
.chat-container {
    display: flex; flex-direction: column;
    gap: 1.2rem;
    max-height: 62vh;
    overflow-y: auto;
    padding: 1rem 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(99,102,241,0.3) transparent;
}

/* ── Individual message bubbles ── */
.msg-user {
    display: flex; justify-content: flex-end;
    animation: slideInRight 0.25s ease-out;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 0.85rem 1.15rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
}
.msg-ai {
    display: flex; justify-content: flex-start; gap: 0.75rem;
    animation: slideInLeft 0.25s ease-out;
}
.msg-ai .avatar {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1.5px solid rgba(99,102,241,0.4);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0; margin-top: 2px;
}
.msg-ai .bubble {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2);
    color: #e2e8f0;
    padding: 0.85rem 1.15rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}

/* ── Source citation cards ── */
.sources-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #818cf8;
    margin: 0.6rem 0 0.3rem 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.source-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 0.28rem 0.65rem;
    font-size: 0.73rem;
    color: #a5b4fc;
    margin: 0.2rem 0.2rem 0 0;
    cursor: default;
}

/* ── Chat input bar ── */
.stChatInput {
    border-top: 1px solid rgba(99,102,241,0.12) !important;
    padding-top: 0.8rem !important;
    background: transparent !important;
}

/* Make Streamlit's outer container transparent so we can style the input box */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
}

/* Style the actual input box container (wraps textarea and button) */
[data-testid="stChatInput"] > div {
    background: rgba(17, 24, 39, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}

/* Add custom focus effect to the container when the user clicks inside */
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(99, 102, 241, 0.7) !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.25) !important;
}

/* Make all internal wrapper divs transparent and borderless */
[data-testid="stChatInput"] > div div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Make the internal textarea transparent and borderless */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}

/* Style the send button inside to match our theme */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    transition: transform 0.1s ease, opacity 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    opacity: 0.9 !important;
    transform: scale(1.05) !important;
}

[data-testid="stChatInput"] button:active {
    transform: scale(0.95) !important;
}

/* ── Hero title section ── */
.hero {
    text-align: center;
    padding: 1.8rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 20%, #c084fc 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #374151;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state h3 { color: #4b5563; font-size: 1.1rem; font-weight: 600; }
.empty-state p  { color: #6b7280; font-size: 0.88rem; margin-top: 0.3rem; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.12);
    margin: 1rem 0;
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #818cf8;
    border-radius: 50%;
    margin: 0 2px;
    animation: pulse 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Info callout ── */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  ── initialize on first load
# ─────────────────────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "rag_chain"   not in st.session_state: st.session_state.rag_chain   = None
if "db_ready"    not in st.session_state: st.session_state.db_ready    = False
if "api_key_ok"  not in st.session_state: st.session_state.api_key_ok  = False

DB_PATH  = r"u:\Projects\University-RAG-Assistant\vector_db\faiss_index"
PDF_DIR  = r"u:\Projects\University-RAG-Assistant\data\pdfs"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def status_badge(label: str, kind: str = "ok") -> str:
    icons = {"ok": "●", "warn": "◐", "err": "○"}
    return f'<span class="status-badge status-{kind}">{icons.get(kind,"●")} {label}</span>'

def check_db_exists() -> bool:
    return os.path.exists(os.path.join(DB_PATH, "index.faiss"))

def load_pipeline(api_key: str):
    """Load or create the FAISS index and wire up the RAG chain."""
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = get_embedding_model()

    if check_db_exists():
        vdb = load_vector_store(DB_PATH, embeddings)
    else:
        return None  # caller must index first

    retriever  = get_retriever(vdb, search_type="mmr", k=3)
    llm        = get_gemini_llm(model_name="gemini-2.5-flash", temperature=0.2)
    chain      = create_rag_chain(retriever, llm)
    return chain

def index_pdfs(api_key: str, uploaded_files):
    """Save uploaded PDFs and build a fresh FAISS index."""
    os.environ["GOOGLE_API_KEY"] = api_key
    os.makedirs(PDF_DIR, exist_ok=True)

    for uf in uploaded_files:
        dest = os.path.join(PDF_DIR, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    all_chunks = []
    for pdf_file in pdf_files:
        docs   = load_pdf(os.path.join(PDF_DIR, pdf_file))
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
        all_chunks.extend(chunks)

    embeddings = get_embedding_model()
    create_vector_store(all_chunks, embeddings, DB_PATH)
    retriever  = get_retriever(
        load_vector_store(DB_PATH, embeddings), search_type="mmr", k=3
    )
    llm   = get_gemini_llm(model_name="gemini-2.5-flash", temperature=0.2)
    chain = create_rag_chain(retriever, llm)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🎓</div>
        <div class="logo-title">UniRAG Assistant</div>
        <div class="logo-sub">AI-Powered Course Guide</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key Detection (Securely Loaded from Environment Only) ──────────────
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key and "your_gemini" not in api_key:
        st.session_state.api_key_ok = True
    else:
        st.session_state.api_key_ok = False
        st.error("🔑 API Key Missing: Please configure `GOOGLE_API_KEY` in your `.env` file.")

    # ── Document Upload ───────────────────────────────────────────────────────
    st.markdown("#### 📂 Upload Course Documents")
    st.markdown('<div class="info-box">Drop your syllabus, lecture notes, or course PDFs here. The assistant will index them for semantic search.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f'{status_badge(f"{len(uploaded_files)} file(s) selected", "warn")}', unsafe_allow_html=True)

    if st.button("⚡ Build Knowledge Base", disabled=not st.session_state.api_key_ok):
        if not uploaded_files and not check_db_exists():
            st.error("Please upload at least one PDF first.")
        else:
            files_to_index = uploaded_files if uploaded_files else None
            with st.spinner("🔄 Embedding documents and building FAISS index…"):
                try:
                    if files_to_index:
                        chain = index_pdfs(api_key, files_to_index)
                    else:
                        chain = load_pipeline(api_key)
                    st.session_state.rag_chain = chain
                    st.session_state.db_ready  = True
                    st.success("✅ Knowledge base ready!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── DB Status ─────────────────────────────────────────────────────────────
    st.markdown("#### 🗄️ System Status")
    if st.session_state.api_key_ok:
        st.markdown(status_badge("API Key: Configured", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("API Key: Missing (.env)", "err"), unsafe_allow_html=True)

    if check_db_exists():
        st.markdown(status_badge("FAISS Index: Loaded", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("FAISS Index: Not Built", "warn"), unsafe_allow_html=True)

    if st.session_state.rag_chain:
        st.markdown(status_badge("RAG Chain: Active", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("RAG Chain: Inactive", "warn"), unsafe_allow_html=True)

    # ── Auto-load if DB already exists ───────────────────────────────────────
    if (check_db_exists()
            and st.session_state.rag_chain is None
            and st.session_state.api_key_ok):
        try:
            chain = load_pipeline(api_key)
            if chain:
                st.session_state.rag_chain  = chain
                st.session_state.db_ready   = True
        except Exception:
            pass

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; font-size:0.72rem; color:#374151; padding-top:1rem;">
        Built with LangChain · FAISS · Gemini · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 University RAG Assistant</h1>
    <p>Ask anything about your course syllabus, policies, lectures, and assignments.</p>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">📄 {s["file"]} · p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            sources_html = f'<div class="sources-header">Sources</div>{chips}'

        st.markdown(f"""
        <div class="msg-ai">
            <div class="avatar">🤖</div>
            <div class="bubble">
                {msg["content"]}
                {sources_html}
            </div>
        </div>""", unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.db_ready and not check_db_exists():
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📚</div>
            <h3>No knowledge base loaded yet</h3>
            <p>Upload your course PDFs in the sidebar and click <b>Build Knowledge Base</b> to get started.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <h3>Ready to answer your questions</h3>
            <p>Try asking about grading policies, deadlines, lecture topics, or assignment details.</p>
        </div>""", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask about your course...",
    disabled=(st.session_state.rag_chain is None)
)

if prompt:
    # 1. Save & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
    <div class="msg-user">
        <div class="bubble">{prompt}</div>
    </div>""", unsafe_allow_html=True)

    # 2. Show typing indicator while waiting
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="msg-ai">
        <div class="avatar">🤖</div>
        <div class="bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    </div>""", unsafe_allow_html=True)

    # 3. Invoke RAG chain
    try:
        response = st.session_state.rag_chain.invoke({"input": prompt})
        answer   = response.get("answer", "Sorry, I could not generate a response.")
        context  = response.get("context", [])

        # Build source list (deduplicated)
        seen    = set()
        sources = []
        for doc in context:
            fname = os.path.basename(doc.metadata.get("source", "Unknown"))
            page  = doc.metadata.get("page", 0) + 1
            key   = (fname, page)
            if key not in seen:
                seen.add(key)
                sources.append({"file": fname, "page": page})

        # 4. Clear typing indicator and render real response
        typing_placeholder.empty()

        chips_html = ""
        if sources:
            chips = "".join(
                f'<span class="source-chip">📄 {s["file"]} · p.{s["page"]}</span>'
                for s in sources
            )
            chips_html = f'<div class="sources-header">Sources</div>{chips}'

        st.markdown(f"""
        <div class="msg-ai">
            <div class="avatar">🤖</div>
            <div class="bubble">
                {answer}
                {chips_html}
            </div>
        </div>""", unsafe_allow_html=True)

        # 5. Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    except Exception as e:
        typing_placeholder.empty()
        err_html = f"⚠️ An error occurred: <code>{e}</code>"
        st.markdown(f"""
        <div class="msg-ai">
            <div class="avatar">🤖</div>
            <div class="bubble" style="border-color:rgba(239,68,68,0.3);">{err_html}</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant", "content": err_html, "sources": []
        })
