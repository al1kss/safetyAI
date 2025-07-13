from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import requests
import json
import os
import zipfile
import hashlib
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import asyncio
from datetime import datetime, timedelta
import jwt

# Simple CloudflareWorker (same as your original)
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
            
            # Load custom knowledge if exists
            knowledge_file = Path(self.data_dir) / "knowledge.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chunks' in data:
                        self.chunks = data['chunks']
            
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
            if isinstance(chunk, str):
                if any(word in chunk.lower() for word in query_lower.split()):
                    results.append(chunk)
            elif isinstance(chunk, dict) and 'content' in chunk:
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

# Multi-user knowledge manager (simplified without LightRAG)
class MultiUserKnowledgeManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.user_stores = {}
    
    def get_user_store(self, user_id: str, ai_id: str = "default") -> SimpleKnowledgeStore:
        store_key = f"{user_id}_{ai_id}"
        if store_key not in self.user_stores:
            user_dir = self.base_dir / f"user_{user_id}" / f"ai_{ai_id}"
            user_dir.mkdir(parents=True, exist_ok=True)
            self.user_stores[store_key] = SimpleKnowledgeStore(str(user_dir))
        return self.user_stores[store_key]
    
    def create_custom_ai(self, user_id: str, ai_name: str, uploaded_files: List[str]) -> str:
        """Create a custom AI from uploaded files"""
        ai_id = str(uuid.uuid4())
        ai_dir = self.base_dir / f"user_{user_id}" / f"ai_{ai_id}"
        ai_dir.mkdir(parents=True, exist_ok=True)
        
        # Process uploaded files into knowledge base
        knowledge_chunks = []
        for file_path in uploaded_files:
            if Path(file_path).exists():
                try:
                    content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                    # Simple chunking - split by paragraphs and sentences
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            # Further split long paragraphs by sentences
                            sentences = para.split('. ')
                            if len(sentences) > 3:
                                # Group sentences into chunks of 3
                                for i in range(0, len(sentences), 3):
                                    chunk = '. '.join(sentences[i:i+3])
                                    if chunk.strip():
                                        knowledge_chunks.append(chunk.strip())
                            else:
                                knowledge_chunks.append(para.strip())
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Save processed knowledge
        knowledge_file = ai_dir / "knowledge.json"
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump({
                "ai_id": ai_id,
                "name": ai_name,
                "chunks": knowledge_chunks,
                "created_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        # Create knowledge store for this AI
        self.user_stores[f"{user_id}_{ai_id}"] = SimpleKnowledgeStore(str(ai_dir))
        
        return ai_id

# Configuration
CLOUDFLARE_API_KEY = os.getenv('CLOUDFLARE_API_KEY', 'lMbDDfHi887AK243ZUenm4dHV2nwEx2NSmX6xuq5')
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/07c4bcfbc1891c3e528e1c439fee68bd/ai/run/"
LLM_MODEL = "@cf/meta/llama-3.2-3b-instruct"
WORKING_DIR = "./dickens"
USER_DATA_DIR = "./user_data"
JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-this')

# Initialize FastAPI
app = FastAPI(title="YourAI Multi-Model API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
cloudflare_worker = None
fire_safety_store = None
user_knowledge_manager = None
users_db: Dict[str, dict] = {}
user_ais: Dict[str, List[dict]] = {}

# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    name: str

class UserLogin(BaseModel):
    email: EmailStr

class QuestionRequest(BaseModel):
    question: str
    mode: str = "hybrid"

class CustomAIRequest(BaseModel):
    name: str
    description: str

class QuestionResponse(BaseModel):
    answer: str
    mode: str
    status: str

class FileUploadResponse(BaseModel):
    filename: str
    size: int
    message: str

# Helper functions
def create_jwt_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data["id"],
        "email": user_data["email"],
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_jwt_token(token)
    user_id = payload["user_id"]
    
    if user_id not in [user["id"] for user in users_db.values()]:
        raise HTTPException(status_code=401, detail="User not found")
    
    return next(user for user in users_db.values() if user["id"] == user_id)

def hash_email(email: str) -> str:
    return hashlib.md5(email.encode()).hexdigest()[:12]

# Initialize system
async def initialize_system():
    global cloudflare_worker, fire_safety_store, user_knowledge_manager
    
    print("🔄 Initializing YourAI System...")
    
    # Initialize Cloudflare worker
    cloudflare_worker = CloudflareWorker(
        cloudflare_api_key=CLOUDFLARE_API_KEY,
        api_base_url=API_BASE_URL,
        llm_model_name=LLM_MODEL,
    )
    
    # Initialize fire safety knowledge store (from existing dickens data)
    dickens_path = Path(WORKING_DIR)
    has_data = dickens_path.exists() and len(list(dickens_path.glob("*.json"))) > 0
    
    if not has_data:
        print("📥 Downloading RAG database...")
        try:
            # Use the same download logic as your original app.py
            data_url = "https://github.com/YOUR_USERNAME/fire-safety-ai/releases/download/v1.0-data/dickens.zip"
            
            print(f"Downloading from: {data_url}")
            response = requests.get(data_url, timeout=60)
            response.raise_for_status()
            
            with open("dickens.zip", "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile("dickens.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.remove("dickens.zip")
            print("Data downloaded!")
            
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            os.makedirs(WORKING_DIR, exist_ok=True)
    
    fire_safety_store = SimpleKnowledgeStore(WORKING_DIR)
    
    # Initialize user knowledge manager
    user_knowledge_manager = MultiUserKnowledgeManager(USER_DATA_DIR)
    
    print("YourAI System ready!")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    await initialize_system()

@app.get("/")
async def root():
    return {"message": "YourAI Multi-Model API", "status": "running", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": ["fire-safety", "general", "physics", "custom"],
        "users_count": len(users_db),
        "active_custom_ais": sum(len(ais) for ais in user_ais.values()),
        "fire_safety_chunks": len(fire_safety_store.chunks) if fire_safety_store else 0
    }

# Authentication endpoints
@app.post("/auth/register")
async def register_user(user_data: UserRegister):
    user_id = hash_email(user_data.email)
    
    if user_data.email in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "created_at": datetime.now().isoformat()
    }
    
    users_db[user_data.email] = user
    user_ais[user_id] = []
    
    token = create_jwt_token(user)
    
    return {
        "user": user,
        "token": token,
        "message": "User registered successfully"
    }

@app.post("/auth/login")
async def login_user(login_data: UserLogin):
    if login_data.email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[login_data.email]
    token = create_jwt_token(user)
    
    return {
        "user": user,
        "token": token,
        "message": "Login successful"
    }

# File upload for custom AI
@app.post("/upload-files", response_model=List[FileUploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]
    user_upload_dir = Path(USER_DATA_DIR) / f"user_{user_id}" / "uploads"
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        if not file.filename:
            continue
        
        # Validate file type
        allowed_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )
        
        # Save file
        file_path = user_upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_files.append(FileUploadResponse(
            filename=file.filename,
            size=file_path.stat().st_size,
            message="Uploaded successfully"
        ))
    
    return uploaded_files

# Create custom AI
@app.post("/create-custom-ai")
async def create_custom_ai(
    ai_data: CustomAIRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]
    user_upload_dir = Path(USER_DATA_DIR) / f"user_{user_id}" / "uploads"
    
    if not user_upload_dir.exists() or not list(user_upload_dir.glob("*")):
        raise HTTPException(status_code=400, detail="No files uploaded. Please upload knowledge files first.")
    
    # Get all uploaded files
    uploaded_files = [str(f) for f in user_upload_dir.glob("*") if f.is_file()]
    
    # Create the custom AI
    ai_id = user_knowledge_manager.create_custom_ai(user_id, ai_data.name, uploaded_files)
    
    # Store AI metadata
    ai_info = {
        "id": ai_id,
        "name": ai_data.name,
        "description": ai_data.description,
        "created_at": datetime.now().isoformat(),
        "files_count": len(uploaded_files)
    }
    
    user_ais[user_id].append(ai_info)
    
    return {
        "ai_id": ai_id,
        "message": "Custom AI created successfully",
        "ai_info": ai_info
    }

# Get user's custom AIs
@app.get("/my-ais")
async def get_user_ais(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    return {
        "ais": user_ais.get(user_id, []),
        "count": len(user_ais.get(user_id, []))
    }

# Chat endpoints for different models
@app.post("/chat/fire-safety", response_model=QuestionResponse)
async def chat_fire_safety(request: QuestionRequest):
    if not cloudflare_worker or not fire_safety_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        print(f"🔥 Fire Safety AI processing: {request.question}")
        
        # Search for relevant context in fire safety knowledge
        relevant_chunks = fire_safety_store.search(request.question, limit=3)
        context = "\n".join(relevant_chunks) if relevant_chunks else "No specific context found."
        
        system_prompt = """You are a Fire Safety AI Assistant specializing in fire safety regulations. 
        Use the provided context to answer questions about building codes, emergency exits, and fire safety requirements."""
        
        user_prompt = f"""Context: {context}

Question: {request.question}

Please provide a helpful answer based on the context about fire safety regulations."""
        
        response = await cloudflare_worker.query(user_prompt, system_prompt)
        return QuestionResponse(answer=response, mode=request.mode, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/general", response_model=QuestionResponse)
async def chat_general(request: QuestionRequest):
    if not cloudflare_worker:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system_prompt = """You are a helpful general AI assistant. Provide accurate, helpful, and engaging responses to user questions."""
    
    try:
        response = await cloudflare_worker.query(request.question, system_prompt)
        return QuestionResponse(answer=response, mode=request.mode, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/custom/{ai_id}", response_model=QuestionResponse)
async def chat_custom_ai(
    ai_id: str,
    request: QuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    if not cloudflare_worker or not user_knowledge_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    user_id = current_user["id"]
    
    # Find the AI
    user_ai_list = user_ais.get(user_id, [])
    ai_info = next((ai for ai in user_ai_list if ai["id"] == ai_id), None)
    
    if not ai_info:
        raise HTTPException(status_code=404, detail="Custom AI not found")
    
    try:
        # Get the knowledge store for this custom AI
        custom_store = user_knowledge_manager.get_user_store(user_id, ai_id)
        
        # Search for relevant context
        relevant_chunks = custom_store.search(request.question, limit=3)
        context = "\n".join(relevant_chunks) if relevant_chunks else "No specific context found."
        
        system_prompt = f"""You are {ai_info['name']}, a custom AI assistant. {ai_info['description']}
        Use the provided context from the uploaded knowledge base to answer questions accurately."""
        
        user_prompt = f"""Context: {context}

Question: {request.question}

Please provide a helpful answer based on the uploaded knowledge base."""
        
        response = await cloudflare_worker.query(user_prompt, system_prompt)
        return QuestionResponse(answer=response, mode=request.mode, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Legacy endpoints (for compatibility with your existing frontend)
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Legacy endpoint that routes to fire safety chat"""
    return await chat_fire_safety(request)

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
