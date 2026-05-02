import os
import streamlit as st
from groq import Groq

class RAGChain:
    def __init__(self, retriever, model: str = "llama-3.3-70b-versatile"):
        self.retriever = retriever
        self.model = model
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

    def ask(self, question: str, top_k: int = 3) -> str:
        context = self.retriever.get_context(question, top_k=top_k)
        context = context[:3000]

        prompt = f"""Answer the question based on the following context.
If the information is not in the context, say "I could not find the information."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content