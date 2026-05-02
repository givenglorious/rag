import os
from src.loader import load_data
from src.embedding import embedding_manage
from vector_store import VectorStore
from retriever import Retriever
from chain import RAGChain

FAISS_STORE_PATH = "faiss_store"

def build_index(pdf_path: str):
    print("\n=== BUILDING INDEX ===")
    
    document = load_data(pdf_path)
    
    embedder = embedding_manage()
    chunks = embedder.check_document(document)
    embeddings = embedder.embed_chunks(chunks)
    
    store = VectorStore(dimension=embeddings.shape[1])
    store.add_embeddings(embeddings, chunks)
    store.save(FAISS_STORE_PATH)
    
    print("\n✅ Index is saved")
    return store, chunks

def load_index():
    print("\n=== LOADING INDEX ===")
    store = VectorStore()
    store.load(FAISS_STORE_PATH)
    return store

def main():
    pdf_path = "data/data_testing_rag_cecepretran.pdf"

    if not os.path.exists(FAISS_STORE_PATH):
        store, _ = build_index(pdf_path)
    else:
        store = load_index()

    retriever = Retriever(store)
    chain = RAGChain(retriever)

    print("\n=== RAG CHATBOT ===")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ["quit", "exit", ""]:
            print("Bye!")
            break
        
        answer = chain.ask(question)
        print(f"\nAnswer: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
