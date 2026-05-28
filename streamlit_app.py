import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime
import uuid
import nltk
import bcrypt

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

load_dotenv()
mongo_client = MongoClient(
    os.getenv("MONGO_URI")
)

db = mongo_client["rag_chatbot"]
users_collection = db["users"]
chat_collection = db["chat_history"]
conversation_collection = db["conversations"]

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">

<style>

/* ── ROOT TOKENS ── */
:root {
    --bg-base:        #080C14;
    --bg-surface:     #0D1220;
    --bg-elevated:    #121929;
    --bg-glass:       rgba(18, 25, 41, 0.72);
    --border:         rgba(255,255,255,0.06);
    --border-accent:  rgba(245, 185, 66, 0.35);
    --accent:         #F5B942;
    --accent-dim:     rgba(245, 185, 66, 0.12);
    --accent-glow:    rgba(245, 185, 66, 0.25);
    --text-primary:   #EEF0F6;
    --text-secondary: #7A8399;
    --text-muted:     #3D4557;
    --user-bubble:    #14213B;
    --ai-bubble:      #0E1A2D;
    --success:        #2DCA8C;
    --danger:         #E05C6B;
    --font-display:   'Syne', sans-serif;
    --font-mono:      'DM Mono', monospace;
}

/* ── GLOBAL RESET ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1100px;
}

/* ── HERO HEADER ── */
.rag-hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.rag-hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -40px;
    width: 340px; height: 200px;
    background: radial-gradient(ellipse, rgba(245,185,66,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.rag-hero-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.rag-hero-title {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem;
}

.rag-hero-title span {
    color: var(--accent);
}

.rag-hero-sub {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 300;
    letter-spacing: 0.02em;
    margin: 0;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── METRICS STRIP ── */
.metric-strip {
    display: flex;
    gap: 1px;
    margin: 1.2rem 0 1.8rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--border);
}

.metric-cell {
    flex: 1;
    background: var(--bg-elevated);
    padding: 0.85rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.metric-val {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}

.metric-key {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── UPLOAD ZONE ── */
div.stFileUploader {
    background: var(--bg-elevated) !important;
    border: 1.5px dashed var(--border-accent) !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s, background 0.2s;
}

div.stFileUploader:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

div.stFileUploader label {
    font-family: var(--font-mono) !important;
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}

/* ── CHAT MESSAGES ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stChatMessageContent"] {
    font-family: var(--font-mono) !important;
    font-size: 0.875rem !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-role="user"] [data-testid="stChatMessageContent"] {
    background: var(--user-bubble) !important;
    border: 1px solid rgba(245, 185, 66, 0.15) !important;
    border-radius: 12px 4px 12px 12px !important;
    padding: 0.85rem 1.1rem !important;
}

/* AI bubble */
[data-testid="stChatMessage"][data-role="assistant"] [data-testid="stChatMessageContent"] {
    background: var(--ai-bubble) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px 12px 12px 12px !important;
    padding: 0.85rem 1.1rem !important;
}

/* Avatar */
[data-testid="stChatMessageAvatar"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}

/* ── BUTTONS ── */
div.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 0.55rem 1rem !important;
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.18s ease !important;
}

div.stButton > button:hover {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    border-color: var(--border-accent) !important;
}

div.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
}

/* Sidebar logo block */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.2rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}

.sidebar-brand-icon {
    width: 32px; height: 32px;
    background: var(--accent-dim);
    border: 1px solid var(--border-accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}

.sidebar-brand-name {
    font-family: var(--font-display) !important;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

.sidebar-brand-ver {
    font-size: 0.6rem;
    color: var(--accent);
    letter-spacing: 0.1em;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    transition: border-color 0.18s;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-base) !important;
    box-shadow: 0 0 10px var(--accent-glow) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: var(--bg-elevated) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:last-child {
    background: var(--accent) !important;
}

/* ── ALERTS & STATUS ── */
.stSuccess {
    background: rgba(45, 202, 140, 0.08) !important;
    border: 1px solid rgba(45, 202, 140, 0.25) !important;
    border-radius: 8px !important;
    color: var(--success) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

.stError {
    background: rgba(224, 92, 107, 0.08) !important;
    border: 1px solid rgba(224, 92, 107, 0.25) !important;
    border-radius: 8px !important;
    color: var(--danger) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

.stWarning {
    background: rgba(245, 185, 66, 0.06) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ── EXPANDER (Retrieved Context) ── */
.streamlit-expanderHeader {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-secondary) !important;
    transition: border-color 0.18s;
}

.streamlit-expanderHeader:hover {
    border-color: var(--border-accent) !important;
    color: var(--accent) !important;
}

.streamlit-expanderContent {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* Chunk card inside expander */
.chunk-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.65;
}

.chunk-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.4rem;
    flex-wrap: wrap;
}

.chunk-tag {
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--border-accent);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), #FFDA85) !important;
    border-radius: 99px !important;
}

.stProgress > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 99px !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-color: var(--accent) transparent transparent transparent !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── FOOTER ── */
.rag-footer {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    padding: 1.5rem 0 0.5rem;
}

.rag-footer span {
    color: var(--accent);
}

/* ── CONVERSATION HISTORY BUTTONS ── */
section[data-testid="stSidebar"] div.stButton > button {
    text-align: left !important;
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--border-accent) !important;
    color: var(--accent) !important;
}

/* ── CAPTION ── */
.stCaption, caption {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-muted) !important;
}

/* ── MARKDOWN BODY TEXT ── */
.stMarkdown p, .stMarkdown li {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.7 !important;
}

</style>
""", unsafe_allow_html=True)


# ── HERO HEADER ──
st.markdown("""
<div class="rag-hero">
    <div class="rag-hero-label">⬡ Semantic Document Intelligence</div>
    <h1 class="rag-hero-title">PDF <span>RAG</span> Assistant</h1>
    <p class="rag-hero-sub">Upload documents · Ask questions · Retrieve contextual answers with AI-powered semantic search</p>
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ── SIDEBAR BRAND ──
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-icon">📘</div>
    <div>
        <div class="sidebar-brand-name">RAG Assistant</div>
        <div class="sidebar-brand-ver">v1.0 · GROQ + FAISS</div>
    </div>
</div>
""", unsafe_allow_html=True)

auth_mode = st.sidebar.selectbox(
    "Authentication",
    ["Login", "Signup"]
)

if "auth_email" not in st.session_state:
    st.session_state.auth_email = ""

if "auth_password" not in st.session_state:
    st.session_state.auth_password = ""

email = st.sidebar.text_input(
    "Email",
    key="auth_email"
)

password = st.sidebar.text_input(
    "Password",
    type="password",
    key="auth_password"
)

if auth_mode == "Signup":
    if st.sidebar.button("✦ Create Account"):
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            st.sidebar.error("User already exists")
        else:
            hashed_password = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            )
            users_collection.insert_one({"email": email, "password": hashed_password})
            st.sidebar.success("Account created successfully!")
            st.session_state.auth_email = ""
            st.session_state.auth_password = ""
            st.rerun()

if auth_mode == "Login":
    if st.sidebar.button("→ Login"):
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.sidebar.success("Login successful!")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

col_new, col_logout = st.sidebar.columns(2)

with col_new:
    if st.button("＋ New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

with col_logout:
    if st.button("⏻ Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("**⚙ RAG Settings**")

chunk_size = st.sidebar.slider(
    "Semantic Chunk Size",
    min_value=200, max_value=1500, value=500, step=100
)

overlap = st.sidebar.slider(
    "Sentence Overlap",
    min_value=0, max_value=5, value=1, step=1
)

top_k = st.sidebar.slider(
    "Top K Retrievals",
    min_value=1, max_value=10, value=3, step=1
)

if st.session_state.logged_in:
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("**💬 Conversations**")

    conversations = conversation_collection.find(
        {"user_email": st.session_state.user_email}
    ).sort("created_at", -1)

    for convo in conversations:
        if st.sidebar.button(
            f"↳ {convo['title']}",
            key=convo["session_id"]
        ):
            st.session_state.session_id = convo["session_id"]
            st.session_state.messages = []
            previous_messages = chat_collection.find(
                {
                    "user_email": st.session_state.user_email,
                    "session_id": st.session_state.session_id
                }
            ).sort("timestamp", 1)
            for msg in previous_messages:
                st.session_state.messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )
            st.rerun()


if not st.session_state.logged_in:
    st.warning("Please login to continue.")
    st.stop()

if st.session_state.logged_in:
    st.sidebar.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.65rem;color:#2DCA8C;
                border:1px solid rgba(45,202,140,0.2);border-radius:6px;
                padding:0.4rem 0.7rem;margin-top:0.5rem;background:rgba(45,202,140,0.05);">
        ● {st.session_state.user_email}
    </div>
    """, unsafe_allow_html=True)

# Messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    previous_messages = chat_collection.find(
        {
            "user_email": st.session_state.user_email,
            "session_id": st.session_state.session_id
        }
    ).sort("timestamp", 1)
    for msg in previous_messages:
        st.session_state.messages.append(
            {"role": msg["role"], "content": msg["content"]}
        )


# ── FILE UPLOAD ──
st.markdown("""
<div class="section-label">Document Upload</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag and drop your PDFs here — or click to browse",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="visible"
)

# Cached Embedding Model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cached PDF Processing
@st.cache_data
def process_pdfs(uploaded_files, chunk_size, overlap):
    all_chunks = []
    chunk_sources = []
    from nltk.tokenize import sent_tokenize

    def semantic_chunk_text(text, chunk_size=500, overlap=1):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                chunks = semantic_chunk_text(extracted_text, chunk_size=chunk_size, overlap=1)
                for chunk_id, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_sources.append(
                        {
                            "file": uploaded_file.name,
                            "page": page_num + 1,
                            "chunk_id": chunk_id + 1
                        }
                    )
    return all_chunks, chunk_sources


# Main App Logic
if uploaded_files:
    with st.spinner("Processing PDFs and building embeddings..."):
        all_chunks, chunk_sources = process_pdfs(uploaded_files, chunk_size, overlap)
        model = load_model()

        if len(all_chunks) == 0:
            st.error("❌ No text could be extracted from the uploaded PDFs.")
            st.stop()

        embeddings = model.encode(all_chunks)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success(f"✅ {len(uploaded_files)} PDF(s) processed and indexed successfully.")

        # Metrics strip
        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric-cell">
                <div class="metric-val">{len(uploaded_files)}</div>
                <div class="metric-key">PDFs Loaded</div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">{len(all_chunks)}</div>
                <div class="metric-key">Text Chunks</div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">{top_k}</div>
                <div class="metric-key">Top-K Retrieval</div>
            </div>
            <div class="metric-cell">
                <div class="metric-val">{chunk_size}</div>
                <div class="metric-key">Chunk Size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    index = None

# ── CHAT SECTION ──
st.markdown("""
<div class="section-label">Conversation</div>
""", unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
query = st.chat_input(
    "Ask a question about your PDFs…",
    disabled=(index is None)
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    chat_collection.insert_one(
        {
            "user_email": st.session_state.user_email,
            "session_id": st.session_state.session_id,
            "role": "user",
            "content": query,
            "timestamp": datetime.utcnow()
        }
    )

    with st.chat_message("user"):
        st.write(query)

    # Query Embedding + Vector Search
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    context = "\n".join([all_chunks[idx] for idx in indices[0]])

    conversation_history = ""
    for message in st.session_state.messages:
        conversation_history += f"{message['role']}: {message['content']}\n"

    existing_conversation = conversation_collection.find_one(
        {"session_id": st.session_state.session_id}
    )
    if not existing_conversation:
        conversation_collection.insert_one(
            {
                "user_email": st.session_state.user_email,
                "session_id": st.session_state.session_id,
                "title": query[:40],
                "created_at": datetime.utcnow()
            }
        )

    prompt = f"""
    You are a helpful PDF assistant.

    Use the conversation history AND retrieved context
    to answer the user's question.

    If the answer is not present in the context,
    say:
    "I could not find that information in the PDFs."

    Conversation History:
    {conversation_history}

    Retrieved Context:
    {context}

    Current Question:
    {query}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    answer = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                answer += delta
                message_placeholder.write(answer)

        # Retrieved Context expander
        with st.expander("📚 Retrieved Context Chunks"):
            for i, idx in enumerate(indices[0]):
                similarity_score = 1 / (1 + distances[0][i])
                metadata = chunk_sources[idx]
                st.markdown(f"""
                <div class="chunk-meta">
                    <span class="chunk-tag">Chunk {i+1}</span>
                    <span class="chunk-tag">📄 {metadata['file']}</span>
                    <span class="chunk-tag">pg {metadata['page']}</span>
                    <span class="chunk-tag">§{metadata['chunk_id']}</span>
                    <span class="chunk-tag">score {similarity_score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(similarity_score))
                st.markdown(f"""
                <div class="chunk-card">{all_chunks[idx]}</div>
                """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    chat_collection.insert_one(
        {
            "user_email": st.session_state.user_email,
            "session_id": st.session_state.session_id,
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow()
        }
    )

# ── FOOTER ──
st.markdown("""
<div class="rag-footer">
    Built with <span>♦</span> using Streamlit · FAISS · MongoDB · Groq LLMs
</div>
""", unsafe_allow_html=True)