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

def get_secret(key):
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
    return val

groq_api_key = get_secret("GROQ_API_KEY")
mongo_uri = get_secret("MONGO_URI")

mongo_client = MongoClient(mongo_uri)
db = mongo_client["rag_chatbot"]
users_collection = db["users"]
chat_collection = db["chat_history"]
conversation_collection = db["conversations"]

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="✦",
    layout="wide"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">

<style>

:root {
    --cream:         #FAF7F2;
    --cream-deep:    #F2EDE4;
    --cream-border:  #E8DFD0;
    --parchment:     #EDE4D6;
    --brown-light:   #C4956A;
    --brown:         #A8703A;
    --brown-dark:    #7A4F28;
    --brown-deeper:  #4E2F10;
    --sienna:        #C4673A;
    --text-dark:     #2C1A0E;
    --text-mid:      #6B4C33;
    --text-soft:     #9C7B5E;
    --text-faint:    #C4AA90;
    --white:         #FFFFFF;
    --shadow-warm:   rgba(122, 79, 40, 0.12);
    --shadow-deep:   rgba(44, 26, 14, 0.18);
    --font-serif:    'Cormorant Garamond', Georgia, serif;
    --font-sans:     'Jost', sans-serif;
}

html, body, .stApp {
    background-color: var(--cream) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-dark) !important;
}

.main .block-container {
    padding: 0 2.5rem 5rem !important;
    max-width: 980px !important;
}

.rag-hero {
    padding: 3rem 0 2rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--cream-border);
    position: relative;
}

.rag-hero-eyebrow {
    font-family: var(--font-sans);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--brown-light);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.rag-hero-eyebrow::before {
    content: '';
    width: 24px;
    height: 1px;
    background: var(--brown-light);
}

.rag-hero-title {
    font-family: var(--font-serif);
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 300;
    font-style: italic;
    line-height: 1.1;
    color: var(--text-dark);
    letter-spacing: -0.01em;
    margin: 0 0 0.75rem;
}

.rag-hero-title strong {
    font-weight: 600;
    font-style: normal;
    color: var(--brown-dark);
}

.rag-hero-sub {
    font-family: var(--font-sans);
    font-size: 0.8rem;
    font-weight: 300;
    color: var(--text-soft);
    letter-spacing: 0.04em;
    line-height: 1.8;
}

.section-rule {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.section-rule-label {
    font-family: var(--font-sans);
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-faint);
    white-space: nowrap;
}

.section-rule-line {
    flex: 1;
    height: 1px;
    background: var(--cream-border);
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--cream-border);
    border: 1px solid var(--cream-border);
    border-radius: 12px;
    overflow: hidden;
    margin: 1.5rem 0 2rem;
}

.metric-tile {
    background: var(--white);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.metric-tile:first-child { border-radius: 11px 0 0 11px; }
.metric-tile:last-child  { border-radius: 0 11px 11px 0; }

.metric-num {
    font-family: var(--font-serif);
    font-size: 2rem;
    font-weight: 600;
    color: var(--brown-dark);
    line-height: 1;
}

.metric-lbl {
    font-family: var(--font-sans);
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-faint);
}

html, body, .stApp {
    background-color: var(--cream) !important;
}

div.stFileUploader {
    background: var(--white) !important;
    border: 1.5px dashed var(--cream-border) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease;
    box-shadow: 0 2px 12px var(--shadow-warm) !important;
}

div.stFileUploader:hover {
    border-color: var(--brown-light) !important;
    box-shadow: 0 4px 20px var(--shadow-deep) !important;
}

div.stFileUploader label p {
    font-family: var(--font-sans) !important;
    color: var(--text-soft) !important;
    font-size: 0.82rem !important;
    font-weight: 300 !important;
}

div.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text-soft) !important;
}

.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
    margin-bottom: 0.25rem !important;
}

[data-testid="stChatMessageContent"] {
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
    line-height: 1.75 !important;
    font-weight: 300 !important;
    color: var(--text-dark) !important;
}

[data-testid="stChatMessage"][data-role="user"] [data-testid="stChatMessageContent"] {
    background: var(--parchment) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 16px 4px 16px 16px !important;
    padding: 0.9rem 1.2rem !important;
    box-shadow: 0 1px 6px var(--shadow-warm) !important;
}

[data-testid="stChatMessage"][data-role="assistant"] [data-testid="stChatMessageContent"] {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 0.9rem 1.2rem !important;
    box-shadow: 0 1px 6px var(--shadow-warm) !important;
}

[data-testid="stChatMessageAvatar"] {
    background: var(--cream-deep) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px var(--shadow-warm) !important;
    transition: all 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--brown-light) !important;
    box-shadow: 0 4px 20px var(--shadow-deep) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
    font-weight: 300 !important;
    color: var(--text-dark) !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-faint) !important;
}

div.stButton > button {
    font-family: var(--font-sans) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    background: var(--white) !important;
    color: var(--text-mid) !important;
    border: 1px solid var(--cream-border) !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 4px var(--shadow-warm) !important;
}

div.stButton > button:hover {
    background: var(--brown-dark) !important;
    color: var(--cream) !important;
    border-color: var(--brown-dark) !important;
    box-shadow: 0 3px 12px var(--shadow-deep) !important;
}

div.stButton > button:active {
    transform: scale(0.98) !important;
}

section[data-testid="stSidebar"] {
    background: var(--cream-deep) !important;
    border-right: 1px solid var(--cream-border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] button {
    font-family: var(--font-sans) !important;
}

.sidebar-brand {
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid var(--cream-border);
    margin-bottom: 1.5rem;
}

.sidebar-brand-title {
    font-family: var(--font-serif) !important;
    font-size: 1.4rem;
    font-weight: 600;
    font-style: italic;
    color: var(--brown-deeper);
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.sidebar-brand-sub {
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-faint);
}

section[data-testid="stSidebar"] label {
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-faint) !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    font-weight: 300 !important;
    box-shadow: 0 1px 4px var(--shadow-warm) !important;
    transition: border-color 0.18s;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--brown-light) !important;
    outline: none !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
    font-size: 0.82rem !important;
    box-shadow: 0 1px 4px var(--shadow-warm) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--brown) !important;
    border: 2px solid var(--cream) !important;
    box-shadow: 0 0 0 3px var(--brown-light) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:last-child {
    background: var(--brown-light) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: var(--cream-border) !important;
}

.stSlider p {
    color: var(--text-mid) !important;
    font-size: 0.8rem !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 300 !important;
}

div[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(168, 112, 58, 0.06) !important;
    border: 1px solid rgba(168, 112, 58, 0.2) !important;
}

.streamlit-expanderHeader {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-radius: 10px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    color: var(--text-soft) !important;
    transition: all 0.18s ease;
    box-shadow: 0 1px 4px var(--shadow-warm) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--brown-light) !important;
    color: var(--brown-dark) !important;
}

.streamlit-expanderContent {
    background: var(--white) !important;
    border: 1px solid var(--cream-border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
}

.chunk-card {
    background: var(--cream) !important;
    border: 1px solid var(--cream-border);
    border-left: 3px solid var(--brown-light);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.78rem;
    font-weight: 300;
    color: var(--text-mid);
    line-height: 1.7;
}

.chunk-meta {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}

.chunk-tag {
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--parchment);
    color: var(--brown);
    border: 1px solid var(--cream-border);
    border-radius: 4px;
    padding: 0.18rem 0.55rem;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--brown-light), var(--sienna)) !important;
    border-radius: 99px !important;
}

.stProgress > div > div {
    background: var(--cream-border) !important;
    border-radius: 99px !important;
    height: 4px !important;
}

.stCaption, caption {
    font-family: var(--font-sans) !important;
    font-size: 0.68rem !important;
    font-weight: 300 !important;
    color: var(--text-faint) !important;
}

.stMarkdown p, .stMarkdown li {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 300 !important;
    color: var(--text-mid) !important;
    line-height: 1.8 !important;
}

section[data-testid="stSidebar"] div.stButton > button {
    text-align: left !important;
    font-size: 0.7rem !important;
    color: var(--text-mid) !important;
}

hr {
    border-color: var(--cream-border) !important;
    margin: 1.25rem 0 !important;
}

.user-pill {
    background: rgba(168, 112, 58, 0.08);
    border: 1px solid rgba(168, 112, 58, 0.2);
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    font-size: 0.65rem;
    font-weight: 400;
    letter-spacing: 0.04em;
    color: var(--brown);
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.rag-footer {
    text-align: center;
    font-family: var(--font-sans);
    font-size: 0.65rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    color: var(--text-faint);
    padding: 2rem 0 0.5rem;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--cream-border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--brown-light); }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rag-hero">
    <div class="rag-hero-eyebrow">Semantic Document Intelligence</div>
    <h1 class="rag-hero-title"><strong>PDF RAG</strong> Assistant</h1>
    <p class="rag-hero-sub">Upload documents &nbsp;·&nbsp; Ask questions &nbsp;·&nbsp; Retrieve contextual answers</p>
</div>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">✦ RAG Assistant</div>
    <div class="sidebar-brand-sub">Groq · FAISS · MongoDB</div>
</div>
""", unsafe_allow_html=True)

auth_mode = st.sidebar.selectbox("Authentication", ["Login", "Signup"])

if "auth_email" not in st.session_state:
    st.session_state.auth_email = ""

if "auth_password" not in st.session_state:
    st.session_state.auth_password = ""

email = st.sidebar.text_input("Email", key="auth_email")
password = st.sidebar.text_input("Password", type="password", key="auth_password")

if auth_mode == "Signup":
    if st.sidebar.button("Create Account"):
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            st.sidebar.error("User already exists")
        else:
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            users_collection.insert_one({"email": email, "password": hashed_password})
            st.sidebar.success("Account created successfully!")
            st.session_state.auth_email = ""
            st.session_state.auth_password = ""
            st.rerun()

if auth_mode == "Login":
    if st.sidebar.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.sidebar.success("Welcome back.")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

st.sidebar.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

col_new, col_logout = st.sidebar.columns(2)

with col_new:
    if st.button("New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

with col_logout:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("**Settings**")

chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=1500, value=500, step=100)
overlap = st.sidebar.slider("Sentence Overlap", min_value=0, max_value=5, value=1, step=1)
top_k = st.sidebar.slider("Top K Retrievals", min_value=1, max_value=10, value=3, step=1)

if st.session_state.logged_in:
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("**Conversations**")

    conversations = conversation_collection.find(
        {"user_email": st.session_state.user_email}
    ).sort("created_at", -1)

    for convo in conversations:
        if st.sidebar.button(f"↳ {convo['title']}", key=convo["session_id"]):
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
    <div class="user-pill">
        ● &nbsp;{st.session_state.user_email}
    </div>
    """, unsafe_allow_html=True)

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

st.markdown("""
<div class="section-rule">
    <span class="section-rule-label">Document Upload</span>
    <span class="section-rule-line"></span>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag and drop your PDFs here — or click to browse",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="visible"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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

if uploaded_files:
    with st.spinner("Processing PDFs…"):
        all_chunks, chunk_sources = process_pdfs(uploaded_files, chunk_size, overlap)
        model = load_model()

        if len(all_chunks) == 0:
            st.error("No text could be extracted from the uploaded PDFs.")
            st.stop()

        embeddings = model.encode(all_chunks)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success(f"{len(uploaded_files)} PDF(s) indexed successfully.")

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-num">{len(uploaded_files)}</div>
                <div class="metric-lbl">PDFs Loaded</div>
            </div>
            <div class="metric-tile">
                <div class="metric-num">{len(all_chunks)}</div>
                <div class="metric-lbl">Text Chunks</div>
            </div>
            <div class="metric-tile">
                <div class="metric-num">{top_k}</div>
                <div class="metric-lbl">Top-K</div>
            </div>
            <div class="metric-tile">
                <div class="metric-num">{chunk_size}</div>
                <div class="metric-lbl">Chunk Size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    index = None

st.markdown("""
<div class="section-rule">
    <span class="section-rule-label">Conversation</span>
    <span class="section-rule-line"></span>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input(
    "Ask a question about your documents…",
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

        with st.expander("View Retrieved Context"):
            for i, idx in enumerate(indices[0]):
                similarity_score = 1 / (1 + distances[0][i])
                metadata = chunk_sources[idx]
                st.markdown(f"""
                <div class="chunk-meta">
                    <span class="chunk-tag">Chunk {i+1}</span>
                    <span class="chunk-tag">{metadata['file']}</span>
                    <span class="chunk-tag">pg {metadata['page']}</span>
                    <span class="chunk-tag">§ {metadata['chunk_id']}</span>
                    <span class="chunk-tag">score {similarity_score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(similarity_score))
                st.markdown(f'<div class="chunk-card">{all_chunks[idx]}</div>', unsafe_allow_html=True)

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

st.markdown("""
<div class="rag-footer">
    Streamlit &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; MongoDB &nbsp;·&nbsp; Groq LLMs
</div>
""", unsafe_allow_html=True)
