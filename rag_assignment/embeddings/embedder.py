from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    """Wrapper for generating embeddings using local SentenceTransformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the model locally. It will automatically download on first run if needed.
        self.model = SentenceTransformer(model_name)
        # Store for reference
        self.model_name = model_name
        
    def get_embedding_dimension(self) -> int:
        """Returns the output dimension of the loaded embedding model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_queries(self, queries: List[str] | str) -> List[List[float]]:
        """
        Embeds a single query or a list of queries. 
        Returns nested list of floats (ready for vector db).
        """
        if isinstance(queries, str):
            queries = [queries]
            
        embeddings = self.model.encode(queries)
        return embeddings.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.
        Returns nested list of floats.
        """
        embeddings = self.model.encode(documents)
        return embeddings.tolist()
