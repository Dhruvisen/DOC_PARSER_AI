from typing import List
from fastapi import APIRouter, UploadFile, File, Form
from app.services.parser_service import route_file
from app.chains.test_chain import run_test_chain
from app.models.message import ChatRequest
from app.services.rag_service import rag_service
from pydantic import BaseModel
from app.agents.analyst_agent import analyst_agent


router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    user_id: str

@router.post("/parse")
async def parse_documents(files: List[UploadFile] = File(...), user_id: str = Form("default")):
    """
    Parse documents and automatically index them into the RAG vector store.
    Supports PDF, Image, DOCX, CSV, Excel, ZIP, Video.
    Returns structured data (summaries + metadata) for each file.
    """
    results = await route_file(files, user_id=user_id)
    return {
        "status": "success",
        "user_id": user_id,
        "files_processed": len(results),
        "data": results
    }

@router.post("/ask-doc")
async def ask_document(request: QuestionRequest):
    """
    Query the uploaded documents using Retrieval-Augmented Generation (RAG).
    Uses the rag_agent to filter by user_id.
    """
    # Using the rag_agent directly for queries as it's the more modern component
    from app.agents.rag_agent import rag_agent
    answer = rag_agent.query(request.question, request.user_id)
    return {"status": "success", "answer": answer}

@router.post("/analyze-data")
async def analyze_data(doc_id: str = Form(...), user_id: str = Form(...), question: str = Form(...)):
    """
    Analyze a previously uploaded CSV or Excel file using its doc_id.
    """
    from app.services.storage_service import storage_service
    file_bytes, file_type = storage_service.get_file(doc_id, user_id)
    
    if not file_bytes:
        return {"status": "error", "message": f"Document ID {doc_id} not found for user {user_id}."}
    
    if file_type not in ["csv", "excel"]:
        return {"status": "error", "message": "Only CSV and Excel documents can be analyzed with this tool."}
    
    result = analyst_agent.analyze(file_bytes, file_type, question)
    return result

@router.post("/clear-index")
async def clear_index():
    """
    Clear the existing RAG index from disk and memory.
    """
    # Both services use 'faiss_index' folder, so clearing it from one works for both
    result = await rag_service.clear_index()
    return {"status": "success", "message": result}
