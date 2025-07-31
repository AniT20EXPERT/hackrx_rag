import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("✅ Basic imports successful")

app = FastAPI()

# Test imports one by one
try:
    import requests
    print("✅ requests imported")
except Exception as e:
    print(f"❌ Failed to import requests: {e}")

try:
    from hash_text import hash_text
    print("✅ hash_text imported")
except Exception as e:
    print(f"❌ Failed to import hash_text: {e}")

try:
    from get_text_pdf import extract_text_from_pdf
    print("✅ get_text_pdf imported")
except Exception as e:
    print(f"❌ Failed to import get_text_pdf: {e}")

try:
    from get_embeddings import split_text
    print("✅ split_text imported")
except Exception as e:
    print(f"❌ Failed to import split_text: {e}")

try:
    from get_embeddings import get_embeddings_batch
    print("✅ get_embeddings_batch imported")
except Exception as e:
    print(f"❌ Failed to import get_embeddings_batch: {e}")

try:
    from llm_answering import answer_question
    print("✅ llm_answering imported")
except Exception as e:
    print(f"❌ Failed to import llm_answering: {e}")

try:
    from pinecone_db import store_embeddings_in_pinecone, query_pinecone_for_context, clear_pinecone_index
    print("✅ pinecone_db imported")
except Exception as e:
    print(f"❌ Failed to import pinecone_db: {e}")

print("🎉 All imports completed")

API_KEY = os.getenv("MY_API_KEY_FOR_AUTH", "test-key-123")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get("/")
async def root():
    return {"message": "FastAPI server is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "imports": "check logs for import status",
        "message": "FastAPI server is running"
    }
