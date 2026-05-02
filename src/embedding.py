from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.loader import load_data

"""
Chunking_size and  chunking_overlap = None
"""

class embedding_manage():
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chunking_overlap: int = 40, chunking_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.chunking_overlap = chunking_overlap
        self.chunking_size = chunking_size
        print(f"Initialized embeddings with model: {model_name}, chunking_overlap: {chunking_overlap}, chunking_size: {chunking_size}") 
    
    def check_document(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_size, 
            chunk_overlap=self.chunking_overlap,

            )
        chunks = text_splitter.split_documents(document)
        print(f"Created {len(chunks)} chunks from the document.")
        return chunks   
    
    def embed_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings for {embeddings.shape[0]} chunks.")
        return embeddings
    
if __name__ == "__main__":
    # Example usage
    
    embedding_instance = embedding_manage()
    document = load_data("data/data_testing_rag_cecepretran.pdf")
    chunks = embedding_instance.check_document(document)
    embeddings = embedding_instance.embed_chunks(chunks)
    
    
    
    
    