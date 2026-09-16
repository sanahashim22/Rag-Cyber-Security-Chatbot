import chromadb
import ollama
import numpy as np

# text
docs = [
    "The Govern (GV) Function establishes and monitors the organization's cybersecurity risk management strategy.",
    "Data Security (PR.DS) ensures that data-at-rest, in-transit, and in-use are managed in accordance with risk strategy."
]

client = chromadb.Client()
collection = client.create_collection(name="test_col", metadata={"hnsw:space": "cosine"})

# embed
for i, d in enumerate(docs):
    res = ollama.embeddings(model="nomic-embed-text", prompt=d)
    collection.add(embeddings=[res["embedding"]], documents=[d], ids=[str(i)])

# query
q = "What does the control PR.DS refer to?"
q_res = ollama.embeddings(model="nomic-embed-text", prompt=q)
results = collection.query(query_embeddings=[q_res["embedding"]], n_results=2)

print("Distances:", results["distances"])
print("Cosine Similarities:", [1.0 - d for d in results["distances"][0]])
