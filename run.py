import os
import uvicorn

def main():
    """
    The main execution entry point for the Self-Healing RAG API.
    
    This script is designed for local development and direct execution.
    In a production environment (like Docker or Kubernetes), you would typically 
    bypass this file and invoke uvicorn directly from the command line:
    `uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4`
    """
    
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    
    environment = os.getenv("ENVIRONMENT", "development").lower()
    reload = environment == "development"

    print(f"🚀 Starting Self-Healing RAG API on http://{host}:{port}")
    print(f"⚙️  Environment: {environment.upper()} | Hot-Reload: {reload}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()