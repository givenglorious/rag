import os
from groq import Groq

class RAGChain:
    def __init__(self, retriever, model: str = "llama3-8b-8192"):
        """
        model pilihan Groq (gratis):
        - llama3-8b-8192
        - llama3-70b-8192
        - mixtral-8x7b-32768
        - gemma2-9b-it
        """
        self.retriever = retriever
        self.model = model
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

  def ask(self, question: str, top_k: int = 3) -> str:
    context = self.retriever.get_context(question, top_k=top_k)
    
    # Batasi panjang context
    context = context[:3000]

    prompt = f"""Jawab pertanyaan berdasarkan konteks berikut.
Jika tidak ada di konteks, katakan "Saya tidak menemukan informasi tersebut."

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:"""

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content