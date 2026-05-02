import streamlit as st
import tempfile
import os

from src.loader import load_data
from src.embedding import embedding_manage
from vector_store import VectorStore
from retriever import Retriever
from chain import RAGChain

st.set_page_config(page_title="RAGChat", page_icon="◈", layout="centered")

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .block-container { max-width: 780px; }
</style>
""", unsafe_allow_html=True)

st.title("◈ RAGChat")
st.caption("Upload PDF, lalu tanya apa saja.")

# ── Session state ──────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar: upload PDF ────────────────────────────────────
with st.sidebar:
    st.header("📄 Dokumen")
    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if uploaded and uploaded.name != st.session_state.doc_name:
        with st.spinner("Memproses dokumen..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                document = load_data(tmp_path)
                embedder = embedding_manage()
                chunks = embedder.check_document(document)
                embeddings = embedder.embed_chunks(chunks)

                store = VectorStore(dimension=embeddings.shape[1])
                store.add_embeddings(embeddings, chunks)

                retriever = Retriever(store)
                st.session_state.chain = RAGChain(retriever)
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"✅ {len(chunks)} chunks siap!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_name:
        st.info(f"📎 {st.session_state.doc_name}")

    st.divider()
    st.markdown("**Model:** Llama 3 · Groq")
    st.markdown("**Embedding:** MiniLM-L12")
    st.markdown("**Vector DB:** FAISS")

# ── Chat ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.chain is None:
    st.info("⬅ Upload PDF dulu dari sidebar untuk mulai.")
else:
    if prompt := st.chat_input("Tanya sesuatu tentang dokumenmu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                answer = st.session_state.chain.ask(prompt)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})