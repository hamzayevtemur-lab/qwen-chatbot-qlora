"""
FastAPI Server for Qwen-2.5-1.5B Chatbot with Real-Time SSE Streaming.
"""
import os
import json
import asyncio
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from app.model_service import ChatModelService

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(512, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.5)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)



# Initialize App & Model Service
app = FastAPI(
    title="Qwen-2.5-1.5B AI Chatbot",
    description="Full-stack fine-tuned QLoRA Chatbot API with real-time SSE streaming.",
    version="1.0.0"
)


# Enable CORS for local testing & cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ChatModelService()

@app.on_event("startup")
async def startup_event():
    """Preloads model on server startup."""
    print("🚀 Starting FastAPI application & warming up model...")
    model_service.load_model()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "device": model_service.device
    }


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams output tokens as SSE JSON events: data: {"token": "..."}
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Convert Pydantic models to standard list of dicts
    formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    async def event_generator():
        try:
            for token in model_service.stream_chat(
                messages=formatted_messages,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                yield {
                    "event": "message",
                    "data": json.dumps({"token": token})
                }

                await asyncio.sleep(0.001)

            # Signal completion
            yield {
                "event": "done",
                "data": json.dumps({"done": True})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    return EventSourceResponse(event_generator())


# Mount Frontend Static Directory
frontend_dir=Path(__file__).parent/"frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")


        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)

    
