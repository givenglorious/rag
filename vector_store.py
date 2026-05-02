import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []  
        print(f"Initialized FAISS index with dimension: {dimension}")

    def add_embeddings(self, embeddings: np.ndarray, chunks: list):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": distances[0][i],
                    "index": idx
                })
        return results

    def save(self, path: str = "faiss_store"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Saved FAISS index to '{path}/'")

    def load(self, path: str = "faiss_store"):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors from '{path}/'")
