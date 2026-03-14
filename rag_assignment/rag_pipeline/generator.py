from typing import List, Dict, Any
from vector_store.endee_client import EndeeDB
from embeddings.embedder import Embedder

# We will use LangChain's generic wrappers. 
# For true production without an OpenAI key, you'd use a local LLM via Ollama or HuggingFace.
# We'll default to a simple demonstration mock if no API key is provided, 
# or use LangChain with OpenAI/DeepSeek if available in the environment.
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, embedder: Embedder, vector_db: EndeeDB):
        self.embedder = embedder
        self.vector_db = vector_db
        
        # Configure LLM. If OPENAI_API_KEY is not set, we'll fall back to a mock answering method
        # for the sake of the assignment being runnable anywhere.
        self.use_mock_llm = not bool(os.environ.get("OPENAI_API_KEY", ""))
        
        if not self.use_mock_llm:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
            
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes the RAG pipeline:
        1. Embed the user question.
        2. Query the vector DB for top-k relevant chunks.
        3. Format the chunks into a context string.
        4. Pass context and question to the LLM (or mock).
        """
        print(f"Embedding query: '{question}'")
        # 1. Embed question 
        # embedder returns nested list, we just want the first vector
        query_embedding = self.embedder.embed_queries(question)[0]
        
        # 2. Retrieve from Endee
        print("Retrieving context from Endee...")
        results = self.vector_db.search(query_embedding=query_embedding, limit=top_k)
        
        # Extract text from results
        context_texts = [res.get("text", "") for res in results]
        
        if not context_texts:
            return {
                "answer": "I don't have enough context in the vector database to answer that question.",
                "context": []
            }
            
        # 3. Format context
        context_string = "\n\n---\n\n".join(context_texts)
        
        print("Generating answer...")
        # 4. Generate Answer
        if self.use_mock_llm:
            # Simple mock response demonstrating the context retrieval worked
            answer = (f"[MOCK LLM] Based on the context provided, here is a synthesized answer. "
                      f"In a real setup with an OPENAI_API_KEY, an LLM would read this context "
                      f"and answer your specific question '{question}'. The top retrieved context was about: "
                      f"'{context_texts[0][:100]}...'")
        else:
            # Use real Langchain + OpenAI pipeline
            prompt = self.prompt_template.format(context=context_string, question=question)
            try:
                response = self.llm.invoke(prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error communicating with LLM: {str(e)}"

        return {
            "answer": answer,
            "context": results
        }
