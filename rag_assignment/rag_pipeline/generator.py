import os
from google import genai
from google.genai import types
from typing import List, Dict, Any
from vector_store.endee_client import EndeeDB
from embeddings.embedder import Embedder
from langchain_core.prompts import PromptTemplate


PROMPT_TEMPLATE = """You are a helpful and polite AI assistant. 

1. If the user greets you or asks who you are, respond politely and mention you can help them analyze their uploaded documents.
2. For factual questions, use only the following retrieved context to answer.
3. If the question requires document context but the answer isn't there, say "I don't have enough context in your documents to answer that."

Keep your answer concise (three sentences maximum).

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, embedder: Embedder, vector_db: EndeeDB):
        self.embedder = embedder
        self.vector_db = vector_db

        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            self.client = genai.Client(api_key=gemini_key)
            self.use_mock_llm = False
            print("✅ Gemini LLM initialized (gemini-flash-latest)")
        else:
            self.client = None
            self.use_mock_llm = True
            print("⚠️  No GEMINI_API_KEY found. Using mock LLM.")

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes the RAG pipeline:
        1. Embed the user question.
        2. Query Endee vector DB for top-k relevant chunks.
        3. Format the chunks into a context string.
        4. Pass context + question to Gemini (or mock).
        """
        print(f"Embedding query: '{question}'")
        query_embedding = self.embedder.embed_queries(question)[0]

        print("Retrieving context from Endee...")
        results = self.vector_db.search(query_embedding=query_embedding, limit=top_k)
        context_texts = [res.get("text", "") for res in results]

        # Prepare prompt
        context_string = "\n\n---\n\n".join(context_texts) if context_texts else "No document context available yet."
        prompt = PROMPT_TEMPLATE.format(context=context_string, question=question)

        # Handle simple greetings even with empty context
        is_greeting = any(word in question.lower() for word in ["hello", "hi ", "hii", "hey", "who are you"])
        
        if not context_texts and not is_greeting:
            return {
                "answer": "I don't have enough context in the vector database to answer. Please upload a document first.",
                "context": []
            }

        print("Generating answer with Gemini...")
        if self.use_mock_llm:
            answer = (
                f"[MOCK LLM] Endee vector search retrieved {len(results)} relevant chunks. "
                f"Add a GEMINI_API_KEY to .env for real AI-generated answers. "
                f"Top retrieved context: '{context_texts[0][:150]}...'"
            )
        else:
            try:
                response = self.client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt,
                )
                answer = response.text
            except Exception as e:
                answer = f"Error communicating with Gemini: {str(e)}"

        return {"answer": answer, "context": results}
