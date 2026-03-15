import os
import uuid
import requests
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# In-memory mock – used when the Endee server is not available locally.
# ---------------------------------------------------------------------------
class _InMemoryStore:
    """A lightweight cosine-similarity store that mirrors the Endee API surface."""

    def __init__(self):
        self._records: List[Dict] = []

    # --- helpers ---
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x * x for x in a) ** 0.5
        nb  = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-10)

    def upsert(self, vectors: List[Dict]):
        for v in vectors:
            self._records.append(v)

    def query(self, vector: List[float], top_k: int = 3) -> List[Dict]:
        scored = [
            {"id": r["id"], "score": self._cosine(vector, r["vector"]), "metadata": r.get("metadata", {})}
            for r in self._records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# EndeeDB – tries the real Endee HTTP API first, falls back to the in-memory mock.
# ---------------------------------------------------------------------------
class EndeeDB:
    """Endee Vector Database Client Wrapper (HTTP + in-memory fallback)."""

    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.host = os.environ.get("ENDEE_HOST", "http://localhost:8080").rstrip("/")
        self.api_key = os.environ.get("ENDEE_API_KEY", "")
        self._mock = None

        # Try to reach the real Endee server.
        try:
            resp = requests.get(f"{self.host}/health", timeout=2)
            resp.raise_for_status()
            print(f"✅ Connected to Endee server at {self.host}")
            self._use_mock = False
            self._ensure_collection_exists()
        except Exception as e:
            print(f"⚠️  Endee server not available ({e}). Using in-memory mock store.")
            self._use_mock = True
            self._mock = _InMemoryStore()

    # --- headers ---
    @property
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _ensure_collection_exists(self):
        try:
            url = f"{self.host}/collections/{self.collection_name}"
            resp = requests.get(url, headers=self._headers, timeout=5)
            if resp.status_code == 404:
                requests.post(
                    f"{self.host}/collections",
                    json={"name": self.collection_name, "dimension": 384, "metric": "cosine"},
                    headers=self._headers,
                    timeout=5,
                ).raise_for_status()
                print(f"Created Endee collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection setup issue: {e}")

    # --- public API ---
    def upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """Upsert document chunks + embeddings into the vector store."""
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            meta = dict(metadatas[i]) if i < len(metadatas) else {}
            meta["text"] = chunk
            vectors.append({"id": str(uuid.uuid4()), "vector": emb, "metadata": meta})

        if self._use_mock:
            self._mock.upsert(vectors)
            print(f"Mock: stored {len(vectors)} vectors.")
            return {"status": "ok", "count": len(vectors)}

        # Real Endee HTTP upsert
        payload = {"vectors": vectors}
        resp = requests.post(
            f"{self.host}/collections/{self.collection_name}/upsert",
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        print(f"Endee: upserted {len(vectors)} vectors.")
        return resp.json()

    def search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Search for the top-k most similar vectors."""
        if self._use_mock:
            raw = self._mock.query(query_embedding, top_k=limit)
        else:
            payload = {"vector": query_embedding, "top_k": limit}
            resp = requests.post(
                f"{self.host}/collections/{self.collection_name}/query",
                json=payload,
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json().get("results", resp.json())

        return [
            {
                "id":       r.get("id", ""),
                "score":    r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
                "text":     r.get("metadata", {}).get("text", ""),
            }
            for r in raw
        ]
