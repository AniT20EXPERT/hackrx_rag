import os
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import requests
from get_text_pdf import extract_text_from_pdf
from get_embeddings import get_embeddings_batch, split_text
from hash_text import hash_text
from llm_answering import answer_question
from pinecone_db import store_embeddings_in_pinecone, query_pinecone_for_context
from pinecone_db import clear_pinecone_index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("FastAPI starting up")

app = FastAPI()

print("FastAPI started")

# Get API key with a default value for testing
API_KEY = os.getenv("MY_API_KEY_FOR_AUTH", "test-key-123")
print(f"API Key loaded: {'Success' if API_KEY else 'Error'}")


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI server is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    print("Health check requested")
    return {
        "status": "healthy",
        "api_key_configured": bool(API_KEY),
        "message": "FastAPI server is running"
    }


# Debug endpoint
@app.get("/debug")
async def debug_info():
    print("Debug info requested")
    return {
        "api_key_set": bool(API_KEY),
        "api_key_length": len(API_KEY) if API_KEY else 0,
        "environment_vars": list(os.environ.keys())
    }


# Simplified version of your endpoint
@app.post("/hackrx/run")
async def run_rag(payload: QueryRequest, authorization: Optional[str] = Header(None)):
    print("Function entered - run_rag")
    
    # Simple auth check
    if not authorization or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # For now, just return a test response
    return {
        "status": "success",
        "message": "Endpoint is working",
        "questions_received": len(payload.questions)
    }


# Add this to ensure the app binds to the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
