import os
from src.loader import load_data
from src.embedding import embedding_manage
from vector_store import VectorStore
from retriever import Retriever
from chain import RAGChain

FAISS_STORE_PATH = "faiss_store"

def build_index(pdf_path: str):
    """Load PDF → chunk → embed → simpan ke FAISS"""
    print("\n=== BUILDING INDEX ===")
    
    # 1. Load PDF
    document = load_data(pdf_path)
    
    # 2. Chunk + Embed
    embedder = embedding_manage()
    chunks = embedder.check_document(document)
    embeddings = embedder.embed_chunks(chunks)
    
    # 3. Simpan ke FAISS
    store = VectorStore(dimension=embeddings.shape[1])
    store.add_embeddings(embeddings, chunks)
    store.save(FAISS_STORE_PATH)
    
    print("\n✅ Index berhasil dibuat dan disimpan!")
    return store, chunks

def load_index():
    """Load FAISS index dari disk"""
    print("\n=== LOADING INDEX ===")
    store = VectorStore()
    store.load(FAISS_STORE_PATH)
    return store

def main():
    pdf_path = "data/data_testing_rag_cecepretran.pdf"

    # Build index kalau belum ada, atau load kalau sudah
    if not os.path.exists(FAISS_STORE_PATH):
        store, _ = build_index(pdf_path)
    else:
        store = load_index()

    # Setup retriever + chain
    retriever = Retriever(store)
    chain = RAGChain(retriever)

    # Chat loop
    print("\n=== RAG CHATBOT SIAP ===")
    print("Ketik 'quit' atau 'exit' untuk keluar.\n")
    
    while True:
        question = input("Pertanyaan: ").strip()
        if question.lower() in ["quit", "exit", ""]:
            print("Bye!")
            break
        
        answer = chain.ask(question)
        print(f"\nJawaban: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
