# main.py - Lightweight version without heavy ML dependencies
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import os
import zipfile
from pathlib import Path
from typing import List


# Simplified CloudflareWorker (no LightRAG dependency)
class CloudflareWorker:
    def __init__(self, cloudflare_api_key: str, api_base_url: str, llm_model_name: str, embedding_model_name: str):
        self.cloudflare_api_key = cloudflare_api_key
        self.api_base_url = api_base_url
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.max_tokens = 4080
        self.max_response_tokens = 4080

    async def _send_request(self, model_name: str, input_: dict):
        headers = {"Authorization": f"Bearer {self.cloudflare_api_key}"}

        try:
            response = requests.post(
                f"{self.api_base_url}{model_name}",
                headers=headers,
                json=input_
            )
            response.raise_for_status()
            result = response.json().get("result", {})

            if "response" in result:
                return result["response"]
            return "Error: No response from Cloudflare"

        except Exception as e:
            print(f"Cloudflare API Error: {e}")
            return f"Error: {e}"

    async def query(self, prompt: str, system_prompt: str = '') -> str:
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_ = {
            "messages": message,
            "max_tokens": self.max_tokens,
        }

        return await self._send_request(self.llm_model_name, input_)


# Simple knowledge store (loads your RAG data)
class SimpleKnowledgeStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunks = []
        self.entities = []
        self.load_data()

    def load_data(self):
        """Load your existing RAG data"""
        try:
            # Load text chunks
            chunks_file = Path(self.data_dir) / "kv_store_text_chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = list(data.values()) if data else []

            # Load entities
            entities_file = Path(self.data_dir) / "vdb_entities.json"
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    self.entities = entities_data.get('data', []) if entities_data else []

            print(f"✅ Loaded {len(self.chunks)} chunks and {len(self.entities)} entities")

        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self.chunks = []
            self.entities = []

    def search(self, query: str, limit: int = 5) -> List[str]:
        """Simple text search through chunks"""
        query_lower = query.lower()
        results = []

        # Search through chunks
        for chunk in self.chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content'].lower()
                if any(word in content for word in query_lower.split()):
                    results.append(chunk['content'])

        # Search through entities
        for entity in self.entities:
            if isinstance(entity, dict):
                entity_text = str(entity).lower()
                if any(word in entity_text for word in query_lower.split()):
                    results.append(str(entity))

        return results[:limit]


# Configuration
CLOUDFLARE_API_KEY = os.getenv('CLOUDFLARE_API_KEY', 'lMbDDfHi887AK243ZUenm4dHV2nwEx2NSmX6xuq5')
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/07c4bcfbc1891c3e528e1c439fee68bd/ai/run/"
EMBEDDING_MODEL = '@cf/baai/bge-m3'
LLM_MODEL = "@cf/meta/llama-3.2-3b-instruct"
WORKING_DIR = "./dickens"

# Initialize FastAPI
app = FastAPI(title="Fire Safety AI Assistant API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
cloudflare_worker = None
knowledge_store = None


class QuestionRequest(BaseModel):
    question: str
    mode: str = "hybrid"


class QuestionResponse(BaseModel):
    answer: str
    mode: str
    status: str


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global cloudflare_worker, knowledge_store

    print("🔄 Initializing Fire Safety AI...")

    # Download data if needed
    dickens_path = Path(WORKING_DIR)
    has_data = dickens_path.exists() and len(list(dickens_path.glob("*.json"))) > 0

    if not has_data:
        print("📥 Downloading RAG database...")
        try:
            # Replace YOUR_USERNAME with your actual GitHub username
            data_url = "https://github.com/al1kss/safetyAI/releases/download/v1.0-data/dickens.zip"

            response = requests.get(data_url, timeout=60)
            response.raise_for_status()

            with open("dickens.zip", "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile("dickens.zip", 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove("dickens.zip")
            print("✅ Data downloaded!")

        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            os.makedirs(WORKING_DIR, exist_ok=True)

    # Initialize components
    cloudflare_worker = CloudflareWorker(
        cloudflare_api_key=CLOUDFLARE_API_KEY,
        api_base_url=API_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
    )

    knowledge_store = SimpleKnowledgeStore(WORKING_DIR)
    print("✅ Fire Safety AI ready!")


@app.get("/")
async def root():
    return {"message": "🔥 Fire Safety AI Assistant API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "knowledge_loaded": len(knowledge_store.chunks) > 0 if knowledge_store else False}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the Fire Safety AI"""

    if not cloudflare_worker or not knowledge_store:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Search for relevant context
        relevant_chunks = knowledge_store.search(request.question, limit=3)

        # Build context
        context = "\n".join(relevant_chunks) if relevant_chunks else "No specific context found."

        # Create prompt
        system_prompt = """You are a Fire Safety AI Assistant specializing in Vietnamese fire safety regulations. 
        Use the provided context to answer questions about building codes, emergency exits, and fire safety requirements."""

        user_prompt = f"""Context: {context}

Question: {request.question}

Please provide a helpful answer based on the context about Vietnamese fire safety regulations."""

        # Get response from Cloudflare
        response = await cloudflare_worker.query(user_prompt, system_prompt)

        return QuestionResponse(
            answer=response,
            mode=request.mode,
            status="success"
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/examples")
async def get_example_questions():
    return {
        "examples": [
            "What are the requirements for emergency exits?",
            "How many exits does a building need?",
            "What are fire safety rules for stairwells?",
            "What are building safety requirements?",
            "What are the fire safety regulations for high-rise buildings?"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)