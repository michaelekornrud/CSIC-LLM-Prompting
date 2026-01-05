import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from models.rag import RAGSystemManager  # NOQA: E402

# Initialize FastAPI app
app = FastAPI(title="Customer Service RAG Chat API")

# Global variable to hold RAG system (lazy loaded)
_rag = None


def get_rag():
    """Lazy load RAG system on first request"""
    global _rag
    if _rag is None:
        print("Loading RAG system...")
        _rag = RAGSystemManager.load(
            system_dir=str(project_root / 'models' / 'rag_system'),
            intent_model_path=str(project_root / 'models' / 'trained' / 'best_intent_classifier.pkl')
        )
        print("✓ RAG system loaded and ready!")
    return _rag


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    customer_message: str
    intent: str
    confidence: float
    response: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Customer Service RAG Chat API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a customer service message through the RAG system
    
    Args:
        request: ChatRequest with customer message
        
    Returns:
        ChatResponse with intent, confidence, and generated response
    """
    print(f"Received message: {request.message}")
    rag = get_rag()
    result = rag.process_message(request.message)
    
    return ChatResponse(
        customer_message=result['customer_message'],
        intent=result['intent'],
        confidence=result['confidence'],
        response=result['response']
    )

