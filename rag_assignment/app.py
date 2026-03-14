import subprocess
import time
import sys
import os

def check_env():
    """Ensure essential directory structure exists."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Copying .env.example to .env...")
        try:
            import shutil
            shutil.copyfile(".env.example", ".env")
        except FileNotFoundError:
            pass

def main():
    print("Starting Endee RAG Application...")
    check_env()
    
    # Start FastAPI backend
    print("Starting FastAPI backend on port 8000...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Wait a moment for backend to initialize before starting frontend
    time.sleep(3)
    
    # Start Streamlit frontend
    print("Starting Streamlit frontend on port 8501...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    try:
        # Keep the script running
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
