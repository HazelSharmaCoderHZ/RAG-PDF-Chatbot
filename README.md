📘 Multi PDF RAG Chatbot

An AI-powered conversational chatbot that allows users to upload multiple PDFs and ask questions using Retrieval-Augmented Generation (RAG).

🛠️ Tech Stack
Category	Tools / Libraries
Frontend	Streamlit
Backend	Python
Embedding Model	sentence-transformers
Vector Database	FAISS
LLM API	Groq API
PDF Processing	pypdf
Numerical Processing	NumPy
Environment Variables	python-dotenv


Features

✅ Upload multiple PDFs
✅ Conversational AI chat interface
✅ Semantic search using embeddings
✅ FAISS vector database retrieval
✅ Multi-document retrieval support
✅ Source chunk visualization
✅ Similarity score display
✅ Streaming AI responses
✅ Conversation memory
✅ Configurable RAG settings
✅ Cached embeddings and processing for better performance


🧠 What I Learned From This Project
This project helped me understand the complete workflow of a real-world RAG (Retrieval-Augmented Generation) system.


Core AI Concepts Learned
🔹 Embeddings
Converted text chunks into vector representations
Learned semantic similarity search
Understood how meaning is represented mathematically
🔹 Chunking
Learned why large documents must be split
Experimented with:
chunk size
overlap
retrieval granularity
🔹 Vector Databases
Used FAISS for semantic retrieval
Learned nearest-neighbor vector search
Understood similarity-based retrieval
🔹 Retrieval-Augmented Generation (RAG)
Combined:
retrieval
embeddings
LLM prompting
Learned how modern AI assistants access external knowledge
🔹 Prompt Engineering
Built prompts using:
retrieved context
conversation history
user query
🔹 Conversational Memory
Implemented session-based memory
Enabled follow-up conversational understanding
🔹 AI System Optimization
Used Streamlit caching
Reduced recomputation overhead
Improved performance for large PDFs
🔹 Explainable AI
Displayed retrieved chunks
Added similarity scores
Improved transparency of AI answers


🧩 Project Architecture: 
PDF Upload
    ->
Text Extraction
    ->
Chunking
    ->
Embeddings Generation
    ->
FAISS Vector Storage
    ->
Semantic Retrieval
    ->
Context Injection
    ->
LLM Response Generation
    ->
Streaming Conversational Output




📚 How Retrieval Works:

PDFs are uploaded
Text is extracted
Text is split into chunks
Chunks are converted into embeddings
Embeddings are stored in FAISS
User query is embedded
Similar chunks are retrieved
Retrieved context is sent to the LLM
AI generates grounded responses
