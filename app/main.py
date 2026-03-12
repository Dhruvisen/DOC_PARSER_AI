"""
AI Document Parser & RAG System
==============================

A high-performance FastAPI backend for multimodal document parsing and Retrieval-Augmented Generation (RAG).

Core Features:
- Multimodal Parsing: Extract text/data from PDF, OCR, Word, Excel, CSV, ZIP, and Video.
- Multi-Tenant RAG: Vector search with user-level isolation using FAISS.
- Agentic Workflows: Purpose-built agents for data analysis, RAG queries, and report writing.
- Cloud-Ready: Support for MinIO (S3-compatible) storage.

Architecture:
- app.api.routes: Fast API endpoints.
- app.services: Core business logic for parsing, storage, and RAG.
- app.agents: Specialized AI agents (Analyst, RAG, Writer).
- loaders: Format-specific data extraction modules.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
import time

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.routes import router as api_router

settings = get_settings()
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Application startup complete.")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced AI Document Parser & RAG System backend.",
    version="1.0.0",
    lifespan=lifespan
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    formatted_process_time = "{0:.2f}ms".format(process_time)
    logger.info(
        f"API Hit: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Time: {formatted_process_time}"
    )
    
    return response

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)