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
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()


def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = chunk_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("RAG System Ready!")


while True:
    query = input("\nAsk a question (or type exit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    k = 3

    distances, indices = index.search(query_embedding, k)

    context = "\n".join([chunks[idx] for idx in indices[0]])

    prompt = f"""
    Answer the question using ONLY the provided context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response.choices[0].message.content

print("\nANSWER:\n")
print(answer)



