import os
from typing import List, Dict, Any
from endee import EndeeClient
from endee.models import Vector, Query
import uuid

class EndeeDB:
    """Endee Vector Database Client Wrapper."""

    def __init__(self, collection_name: str = "rag_collection"):
        # Endee usually initializes with an API key, or locally without one depending on the setup. Let's use the local API host if set, else defaults.
        api_key = os.environ.get("ENDEE_API_KEY", "")
        host = os.environ.get("ENDEE_HOST", "http://localhost:8000") # Default local host assumption. adjust as needed

        print(f"Initializing EndeeClient on {host} for collection '{collection_name}'...")
        # Note: Endee Python SDK documentation might differ slightly on exact initialization. Standardizing on common patterns.
        # Assuming EndeeClient(url, api_key) or similar. 
        # For this assignment, we'll try initializing and create a collection wrapper method.
        # Actually standard python Endee client:
        try:
             self.client = EndeeClient(host=host, api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize Endee Client strictly with host/api_key. Trying default. {e}")
            self.client = EndeeClient()
            
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't already exist."""
        # Note: Endee SDK syntax for creating a collection might vary.
        # Often vector databases want dimension specified. We'll try dynamic or assume 384 (all-MiniLM-L6-v2)
        try:
           collections = self.client.list_collections()
           if self.collection_name not in [c.name for c in collections]:
              print(f"Creating Endee collection: {self.collection_name}")
              self.client.create_collection(
                  name=self.collection_name, 
                  dimension=384, # default for all-MiniLM-L6-v2
                  metric="cosine"
              )
        except Exception as e:
           print(f"Collection check/create encountered an issue (maybe it already exists): {e}")


    def upsert_chunks(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Upserts document chunks and their embeddings into the vector database.
        """
        vectors = []
        for i in range(len(chunks)):
            # Combine chunk text into metadata so we can retrieve it
            meta = metadatas[i] if i < len(metadatas) else {}
            meta["text"] = chunks[i]
            
            # Using UUID for deterministic updates is better, but random uuid is fine for this demo
            vector_id = str(uuid.uuid4())
            
            vectors.append(
                Vector(id=vector_id, vector=embeddings[i], metadata=meta)
            )

        print(f"Upserting {len(vectors)} vectors to Endee...")
        response = self.client.upsert(collection_name=self.collection_name, vectors=vectors)
        return response

    def search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the Endee database for vectors similar to the query_embedding.
        Returns the metadata (which includes the original chunk text).
        """
        query = Query(
            vector=query_embedding,
            top_k=limit
        )
        
        results = self.client.query(collection_name=self.collection_name, query=query)
        
        # Format the results returning the original text and metadata
        formatted_results = []
        if hasattr(results, 'matches'):
            # Endee might return .matches
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "text": match.metadata.get("text", "") if match.metadata else ""
                })
        else:
             # Fallback parsing if results is simply a list
             for match in results:
                 # Assumes match is dict-like
                 metadata = getattr(match, 'metadata', {})
                 formatted_results.append({
                     "score": getattr(match, 'score', 0.0),
                     "text": metadata.get("text", ""),
                     "metadata": metadata
                 })

        return formatted_results
