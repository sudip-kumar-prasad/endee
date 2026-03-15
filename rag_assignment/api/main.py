import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# We look for .env in the same directory as this file's parent (rag_assignment/)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# Local imports
from utils.document_processor import DocumentProcessor
from embeddings.embedder import Embedder
from vector_store.endee_client import EndeeDB
from rag_pipeline.generator import RAGPipeline

app = FastAPI(
    title="Endee RAG API",
    description="API for Document Ingestion and Semantic Search using Endee Vector DB",
    version="1.0.0"
)

# Initialize singletons when app starts (in a real app, use dependency injection/lifespan events)
print("Initializing core components...")
doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
embedder = Embedder()
vector_db = EndeeDB(collection_name="rag_documents")
rag_pipe = RAGPipeline(embedder=embedder, vector_db=vector_db)
print("Initialization complete.")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a PDF or Text file.
    Process: 
    1. Save temporarily 
    2. Read & Chunk
    3. Embed
    4. Store in Endee DB
    """
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process / Chunk
        print(f"Processing uploaded file: {file.filename}")
        chunks = doc_processor.load_and_chunk(temp_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from the file.")
            
        print(f"Generated {len(chunks)} chunks.")
            
        # Extract text and meta
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add original filename to metadata
        for meta in metadatas:
            meta["source_file"] = file.filename
            
        # Embed
        print(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed_documents(texts)
        
        # Store in Endee
        print("Upserting vectors into Endee...")
        vector_db.upsert_chunks(chunks=texts, embeddings=embeddings, metadatas=metadatas)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {file.filename}", 
            "chunks_processed": len(texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Handle a user query to search the Endee database and return an AI answer.
    """
    try:
        result = rag_pipe.generate_answer(
            question=request.question, 
            top_k=request.top_k
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
