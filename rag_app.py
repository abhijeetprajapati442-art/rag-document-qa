import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

def chunk(text, size=800, overlap=100):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

def embed(texts):
    out = []
    try:
        for i in range(0, len(texts), 50):
            res = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i+50])
            out += [e.values for e in res.embeddings]
        return out
    except Exception as e:
        print("EMBED ERROR:", e)
        raise

text = open("notes.txt", encoding="utf-8").read()
chunks = chunk(text)

col = chromadb.Client().create_collection("docs")
col.add(ids=[str(i) for i in range(len(chunks))],
        documents=chunks, embeddings=embed(chunks))

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break
    hits = col.query(query_embeddings=embed([q]), n_results=3)["documents"][0]
    context = "\n---\n".join(hits)
    prompt = (f"Answer using only this context. If the answer isn't there, "
              f"say you don't know.\n\nContext:\n{context}\n\nQuestion: {q}")
    print(client.models.generate_content(model=CHAT_MODEL, contents=prompt).text)