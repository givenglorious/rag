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

    def ask(self, question: str, top_k: int = 5) -> str:
        """Ambil context dari retriever, lalu tanya ke Groq"""
        context = self.retriever.get_context(question, top_k=top_k)

        prompt = f"""Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan.
Jika jawaban tidak ada di dalam konteks, katakan "Saya tidak menemukan informasi tersebut dalam dokumen."

=== KONTEKS ===
{context}

=== PERTANYAAN ===
{question}

=== JAWABAN ==="""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        return answer
