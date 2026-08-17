# ◈ RAGChat
https://r3ofl4cbie5b8jsjegxjhp.streamlit.app/

Chat with your PDF documents using AI. Upload a PDF, ask anything — answers are grounded in your document.

## Tech Stack

- **Frontend/UI** — Streamlit
- **Embedding** — `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers)
- **Vector DB** — FAISS
- **LLM** — Llama 3.3 via Groq API (free)

## Features

- Upload PDF directly from the browser
- Automatic chunking & embedding
- Similarity search with FAISS
- Context-grounded answers (RAG)
- Supports English & Indonesian

## Project Structure

```
rag/
├── streamlit_app.py      # Main UI
├── vector_store.py       # FAISS index
├── retriever.py          # Similarity search
├── chain.py              # Groq LLM chain
├── requirements.txt
└── src/
    ├── loader.py         # PDF loader
    └── embedding.py      # Chunking & embedding
```

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/givenglorious/rag.git
cd rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create `.streamlit/secrets.toml`**
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxx"
```

**4. Run**
```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file: `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxx"
```
5. Deploy!

## Getting a Free Groq API Key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create a new API key
3. Paste it into Streamlit secrets
