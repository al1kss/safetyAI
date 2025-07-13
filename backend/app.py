from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import os
import zipfile
from pathlib import Path
from typing import List
import asyncio

# Simple CloudflareWorker without heavy dependencies
class CloudflareWorker:
    def __init__(self, cloudflare_api_key: str, api_base_url: str, llm_model_name: str):
        self.cloudflare_api_key = cloudflare_api_key
        self.api_base_url = api_base_url
        self.llm_model_name = llm_model_name
        self.max_tokens = 4080

    async def _send_request(self, model_name: str, input_: dict):
        headers = {"Authorization": f"Bearer {self.cloudflare_api_key}"}
        
        try:
            response_raw = requests.post(
                f"{self.api_base_url}{model_name}",
                headers=headers,
                json=input_,
                timeout=30
            ).json()
            
            result = response_raw.get("result", {})
            
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
        
        result = await self._send_request(self.llm_model_name, input_)
        return result

# Simple knowledge store that loads your RAG data
class SimpleKnowledgeStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunks = []
        self.entities = []
        self.load_data()
    
    def load_data(self):
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
                    if isinstance(entities_data, dict) and 'data' in entities_data:
                        self.entities = entities_data['data']
                    elif isinstance(entities_data, list):
                        self.entities = entities_data
                    else:
                        self.entities = []
                    
            print(f"✅ Loaded {len(self.chunks)} chunks and {len(self.entities)} entities")
            
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self.chunks = []
            self.entities = []
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        query_lower = query.lower()
        results = []
        
        # Search through chunks
        for chunk in self.chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content']
                if any(word in content.lower() for word in query_lower.split()):
                    results.append(content)
        
        # Search through entities
        for entity in self.entities:
            if isinstance(entity, dict):
                entity_text = json.dumps(entity, ensure_ascii=False)
                if any(word in entity_text.lower() for word in query_lower.split()):
                    results.append(entity_text)
        
        return results[:limit]

# Configuration
CLOUDFLARE_API_KEY = os.getenv('CLOUDFLARE_API_KEY', 'lMbDDfHi887AK243ZUenm4dHV2nwEx2NSmX6xuq5')
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/07c4bcfbc1891c3e528e1c439fee68bd/ai/run/"
LLM_MODEL = "@cf/meta/llama-3.2-3b-instruct"
WORKING_DIR = "./dickens"

# Initialize FastAPI
app = FastAPI(title="Fire Safety AI Assistant API", version="1.0.0")

# Enable CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
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

async def initialize_system():
    global cloudflare_worker, knowledge_store
    
    print("🔄 Initializing Fire Safety AI...")
    
    # Download data if needed
    dickens_path = Path(WORKING_DIR)
    has_data = dickens_path.exists() and len(list(dickens_path.glob("*.json"))) > 0
    
    if not has_data:
        print("📥 Downloading RAG database...")
        try:
            # REPLACE YOUR_USERNAME with your actual GitHub username
            data_url = "https://github.com/YOUR_USERNAME/fire-safety-ai/releases/download/v1.0-data/dickens.zip"
            
            print(f"Downloading from: {data_url}")
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
        llm_model_name=LLM_MODEL,
    )
    
    knowledge_store = SimpleKnowledgeStore(WORKING_DIR)
    print("✅ Fire Safety AI ready!")

@app.on_event("startup")
async def startup_event():
    await initialize_system()

@app.get("/")
async def root():
    return {"message": "🔥 Fire Safety AI Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "knowledge_loaded": len(knowledge_store.chunks) > 0 if knowledge_store else False,
        "chunks_count": len(knowledge_store.chunks) if knowledge_store else 0,
        "entities_count": len(knowledge_store.entities) if knowledge_store else 0
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not cloudflare_worker or not knowledge_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        print(f"🔍 Processing question: {request.question}")
        
        # Search for relevant context in your RAG data
        relevant_chunks = knowledge_store.search(request.question, limit=3)
        
        # Build context from your fire safety knowledge
        context = "\n".join(relevant_chunks) if relevant_chunks else "No specific context found."
        
        # Create prompt for Cloudflare AI
        system_prompt = """You are a Fire Safety AI Assistant specializing in Vietnamese fire safety regulations. 
        Use the provided context to answer questions about building codes, emergency exits, and fire safety requirements."""
        
        user_prompt = f"""Context: {context}

Question: {request.question}

Please provide a helpful answer based on the context about Vietnamese fire safety regulations."""
        
        # Get response from Cloudflare AI
        response = await cloudflare_worker.query(user_prompt, system_prompt)
        
        return QuestionResponse(
            answer=response,
            mode=request.mode,
            status="success"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/modes")
async def get_available_modes():
    return {
        "modes": [
            {"name": "hybrid", "description": "Combined approach (recommended)"},
            {"name": "local", "description": "Search specific document sections"},
            {"name": "global", "description": "Look at overall document themes"},
            {"name": "naive", "description": "Simple text search"}
        ]
    }

@app.get("/examples")
async def get_example_questions():
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
    uvicorn.run(app, host="0.0.0.0", port=7860)
