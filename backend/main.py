# main.py - FastAPI Backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import os
import requests
import numpy as np
from typing import List


# Your CloudflareWorker class
class CloudflareWorker:
    def __init__(self, cloudflare_api_key: str, api_base_url: str, llm_model_name: str, embedding_model_name: str):
        self.cloudflare_api_key = cloudflare_api_key
        self.api_base_url = api_base_url
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.max_tokens = 4080
        self.max_response_tokens = 4080

    async def _send_request(self, model_name: str, input_: dict, debug_log: str):
        headers = {"Authorization": f"Bearer {self.cloudflare_api_key}"}

        try:
            response_raw = requests.post(
                f"{self.api_base_url}{model_name}",
                headers=headers,
                json=input_
            ).json()

            result = response_raw.get("result", {})

            if "data" in result:
                return np.array(result["data"])
            if "response" in result:
                return result["response"]

            raise ValueError(f"Unexpected response format: {response_raw}")

        except Exception as e:
            print(f"Cloudflare API Error: {e}")
            return None

    async def query(self, prompt: str, system_prompt: str = '', **kwargs) -> str:
        kwargs.pop("hashing_kv", None)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_ = {
            "messages": message,
            "max_tokens": self.max_tokens,
            "response_token_limit": self.max_response_tokens,
        }

        result = await self._send_request(self.llm_model_name, input_, "")
        return result if result is not None else "Error: Failed to get response"

    async def embedding_chunk(self, texts: List[str]) -> np.ndarray:
        input_ = {
            "text": texts,
            "max_tokens": self.max_tokens,
            "response_token_limit": self.max_response_tokens,
        }

        result = await self._send_request(self.embedding_model_name, input_, "")

        if result is None:
            return np.random.rand(len(texts), 1024).astype(np.float32)

        return result


# Configuration
CLOUDFLARE_API_KEY = os.getenv('CLOUDFLARE_API_KEY', 'lMbDDfHi887AK243ZUenm4dHV2nwEx2NSmX6xuq5')
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/07c4bcfbc1891c3e528e1c439fee68bd/ai/run/"
EMBEDDING_MODEL = '@cf/baai/bge-m3'
LLM_MODEL = "@cf/meta/llama-3.2-3b-instruct"
WORKING_DIR = "./dickens"

# Initialize FastAPI
app = FastAPI(title="Fire Safety AI Assistant API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance = None


# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # naive, local, global, hybrid


class QuestionResponse(BaseModel):
    answer: str
    mode: str
    status: str


import zipfile
import urllib.request


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_instance

    print("🔄 Initializing RAG system...")

    # Download data if not exists
    if not os.path.exists(WORKING_DIR):
        print("📥 Downloading RAG data...")
        try:
            # Download from GitHub (replace with your actual repo URL)
            data_url = "https://github.com/al1kss/safetyAI/archive/refs/heads/main.zip"
            urllib.request.urlretrieve(data_url, "data.zip")

            # Extract
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall("./temp")

            # Move dickens folder to correct location
            import shutil
            shutil.move("./temp/fire-safety-data-main/dickens", WORKING_DIR)

            # Cleanup
            os.remove("data.zip")
            shutil.rmtree("./temp")
            print("✅ Data downloaded successfully!")

        except Exception as e:
            print(f"⚠️ Could not download data: {e}")
            print("🔄 Creating new empty data directory...")
            os.makedirs(WORKING_DIR, exist_ok=True)

    cloudflare_worker = CloudflareWorker(
        cloudflare_api_key=CLOUDFLARE_API_KEY,
        api_base_url=API_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
    )

    rag_instance = LightRAG(
        working_dir=WORKING_DIR,
        max_parallel_insert=2,
        llm_model_func=cloudflare_worker.query,
        llm_model_name=LLM_MODEL,
        llm_model_max_token_size=4080,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=2048,
            func=lambda texts: cloudflare_worker.embedding_chunk(texts),
        ),
    )

    await rag_instance.initialize_storages()
    print("✅ RAG system initialized!")


@app.get("/")
async def root():
    return {"message": "🔥 Fire Safety AI Assistant API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_ready": rag_instance is not None}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the Fire Safety AI"""

    if not rag_instance:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Query the RAG system
        print(f"🔍 Processing question: {request.question}")

        response = await rag_instance.aquery(
            request.question,
            param=QueryParam(mode=request.mode)
        )

        return QuestionResponse(
            answer=response,
            mode=request.mode,
            status="success"
        )

    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/modes")
async def get_available_modes():
    """Get available query modes"""
    return {
        "modes": [
            {"name": "naive", "description": "Simple text search"},
            {"name": "local", "description": "Search specific document sections"},
            {"name": "global", "description": "Look at overall document themes"},
            {"name": "hybrid", "description": "Combined approach (recommended)"}
        ]
    }


# Example questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example questions users can ask"""
    return {
        "examples": [
            "What are the requirements for emergency exits?",
            "How many exits does a building need?",
            "What are fire safety rules for stairwells?",
            "What are building safety requirements?",
            "What are the fire safety regulations for high-rise buildings?",
            "What are the requirements for fire doors?",
            "How should evacuation routes be designed?"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)