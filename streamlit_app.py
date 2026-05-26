import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📘",
    layout="wide"
)

st.title("📘 PDF RAG Chatbot")

# Sidebar Settings
st.sidebar.header("⚙️ RAG Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=100,
    max_value=1000,
    value=200,
    step=50
)

overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=300,
    value=50,
    step=10
)

top_k = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# File Upload
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
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

    def chunk_text(text, chunk_size=200, overlap=50):

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            all_chunks.append(chunk)

            chunk_sources.append(uploaded_file.name)

    return all_chunks, chunk_sources

# Main App Logic
if uploaded_files:

    all_chunks, chunk_sources = process_pdfs(
        uploaded_files,
        chunk_size,
        overlap
    )

    model = load_model()

    embeddings = model.encode(all_chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success(f"✅ Processed {len(uploaded_files)} PDFs successfully!")

    # Display Chat History
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    query = st.chat_input("Ask a question about the PDFs")

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.write(query)

        # Query Embedding
        query_embedding = model.encode([query])

        query_embedding = np.array(query_embedding).astype("float32")

        # Vector Search
        distances, indices = index.search(
            query_embedding,
            top_k
        )

        context = "\n".join(
            [all_chunks[idx] for idx in indices[0]]
        )

        conversation_history = ""

        for message in st.session_state.messages:

            role = message["role"]

            content = message["content"]

            conversation_history += f"{role}: {content}\n"

        # Prompt
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

        # Streaming Response
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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

            # Retrieved Context
            with st.expander("📚 View Retrieved Context"):

                for i, idx in enumerate(indices[0]):

                    similarity_score = 1 / (1 + distances[0][i])

                    st.write(f"📄 Chunk {i+1}")

                    st.caption(
                        f"Source: {chunk_sources[idx]}"
                    )

                    st.caption(
                        f"Similarity Score: {similarity_score:.4f}"
                    )

                    st.write(all_chunks[idx])

                    st.divider()

        # Save Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )