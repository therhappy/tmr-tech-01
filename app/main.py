"""
Main application entry point for the Conversational LLM Application.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import the config module (app/__init__.py ensures config is loaded first)
from app.config import get_api_host, get_api_port
from app.llm.conversation import LLMConversation


# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tech practical: Conversational LLM on the FinQA dataset API",
    description="API for interacting with a conversational LLM \
        answering questions on the ConvFinQA dataset, as a technical evaluation \
        for Timon Ther's application to Tomoro.AI",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ConversationRequest(BaseModel):
    message: str
    conversation_id: str = None

class ConversationResponse(BaseModel):
    response: str
    conversation_id: str

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize LLM conversation handler
conversation_handler = LLMConversation()

@app.get("/")
async def root():
    """Return the app landing page"""
    return FileResponse("static/index.html")

@app.post("/conversation", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest):
    """
    Create a conversation or continue an existing one.
    """
    try:
        logger.info(f"Processing conversation request: {request.conversation_id}")
        response, conversation_id = conversation_handler.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
        )
        return ConversationResponse(response=response, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use configuration values from config
    host = get_api_host()
    port = get_api_port()
    logger.info(f"Starting API server at {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
