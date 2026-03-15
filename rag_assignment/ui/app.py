import streamlit as st
import requests
import os
import subprocess
import sys
import time
import socket
from dotenv import load_dotenv

load_dotenv()

# Assuming the FastAPI backend runs on 8000
API_URL = os.getenv("API_URL", "http://localhost:8000")

def is_backend_running(host="localhost", port=8000):
    """Check if the backend port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_backend():
    """Start the FastAPI backend as a subprocess."""
    if not is_backend_running():
        st.info("Backend not detected. Starting FastAPI backend...")
        # Get the path to the backend directory (one level up from ui/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Start uvicorn as a subprocess
        # Note: We use sys.executable to ensure we use the same python interpreter/venv
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=base_dir,
            stdout=subprocess.DEVNULL, # Suppress logs in the UI
            stderr=subprocess.DEVNULL,
            env=os.environ.copy()
        )
        # Wait a few seconds for startup
        time.sleep(5)
        if is_backend_running():
            st.success("Backend started successfully!")
        else:
            st.warning("Backend starting in background. Attempting to connect...")

# Ensure backend is running before any UI interaction
start_backend()

st.set_page_config(
    page_title="Endee RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Endee RAG Assistant")
st.markdown("Upload a PDF or Text document, and then ask questions about it using Semantic Search powered by Endee Vector Database.")

# Sidebar for Upload
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt", "md"])
    
    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner(f"Ingesting '{uploaded_file.name}' into Endee DB..."):
                try:
                    # Send file to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Processed {data.get('chunks_processed', 0)} chunks.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {str(e)}")
        else:
            st.warning("Please select a file first.")

# Main area for chatting/querying
st.header("2. Ask Questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI /query
    with st.spinner("Searching Endee and Generating Answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query", 
                json={"question": prompt, "top_k": 3}
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                answer = data.get("answer", "No answer generated.")
                contexts = data.get("context", [])
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Optionally show retrieved context
                    with st.expander("View Retrieved Context (Endee Vector DB Matches)"):
                        for i, ctx in enumerate(contexts):
                            st.markdown(f"**Match {i+1} (Score: {ctx.get('score', 0):.4f})**")
                            st.write(f"_{ctx.get('text', 'No text')}..._")
                            if "metadata" in ctx and "source_file" in ctx["metadata"]:
                                st.caption(f"Source: {ctx['metadata'].get('source_file')}")
                            st.divider()

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            else:
                st.error(f"Query Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")
