from pathlib import Path
import streamlit as st
import tempfile, os

from src.loader import loader_data
from src.embedding import embedding_manage
from vector_store import VectorStore
from retriever import Retriever
from chain import RAGChain

st.set_page_config(page_title="RAGChat", page_icon="◈", layout="centered")
st.title("◈ RAGChat")
st.caption("Upload your document and ask questions about it!")

for key, val in [("chain", None), ("messages", []), ("doc_name", None)]:
    st.session_state.setdefault(key, val)

with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader(
        "Upload Document",
        type=["pdf", "txt", "docx", "csv", "json"]
    )

    if uploaded and uploaded.name != st.session_state.doc_name:
        with st.spinner("Processing..."):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                docs     = loader_data(tmp_path)
                embedder = embedding_manage()
                chunks   = embedder.check_document(docs)
                embeddings = embedder.embed_chunks(chunks)

                store = VectorStore(dimension=embeddings.shape[1])
                store.add_embeddings(embeddings, chunks)

                st.session_state.chain    = RAGChain(Retriever(store))
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"✅ {len(chunks)} chunks ready!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_name:
        st.info(f"📎 {st.session_state.doc_name}")

    st.divider()
    st.markdown("**Model:** Llama 3 · Groq  \n**Embedding:** MiniLM-L12  \n**Vector DB:** FAISS")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.chain is None:
    st.info("⬅ Upload a document to get started.")
elif prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.ask(prompt)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})