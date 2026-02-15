from fastapi import APIRouter, UploadFile, File
from app.services.parser_service import route_file
from app.chains.test_chain import run_test_chain
from app.models.message import ChatRequest
from app.services.rag_service import rag_service
from pydantic import BaseModel

router = APIRouter()

@router.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse a document using the configured parser agent.
    Supports PDF, Image, DOCX, CSV, Excel, ZIP, Video.
    """
    result = await route_file(file)
    return {
        "status": "success",
        "filename": file.filename,
        "data": result
    }



class QuestionRequest(BaseModel):
    question: str

@router.post("/test-llm")
async def test_llm(request: ChatRequest):
    """
    Test the LLM connection with multimodal support.
    Accepts a list of messages with role and content.
    """
    try:
        response = run_test_chain(request)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF/TXT) for RAG processing.
    """
    result = await rag_service.ingest_document(file)
    return {"status": "success", "message": result}

@router.post("/ask-doc")
async def ask_document(request: QuestionRequest):
    """
    Query the uploaded document using Retrieval-Augmented Generation (RAG).
    """
    answer = await rag_service.query_document(request.question)
    return {"status": "success", "answer": answer}
