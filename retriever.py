from vector_store import VectorStore
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, vector_store: VectorStore, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)
        print(f"Retriever initialized with model: {model_name}")

    def retrieve(self, query: str, top_k: int = 3):
        print(f"Retrieving top-{top_k} chunks for query: '{query}'")
        query_embedding = self.model.encode(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        print(f"Found {len(results)} relevant chunks.")
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        context = "\n\n".join([r["chunk"].page_content for r in results])
        return context

