import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles ingestion and chunking of documents (PDFs and Text)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_chunk(self, file_path: str) -> List[Document]:
        """Loads a file and splits it into chunks."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_text(self, text: str, source_name: str = "text_input") -> List[Document]:
        """Creates Document chunks from raw text."""
        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self.text_splitter.split_documents([doc])
        return chunks
