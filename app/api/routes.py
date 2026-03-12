import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Form
from app.services.parser_service import route_file
from app.chains.test_chain import run_test_chain
from app.models.message import ChatRequest
from app.services.rag_service import rag_service
from pydantic import BaseModel
from app.agents.analyst_agent import analyst_agent
from app.agents.rag_agent import rag_agent
from app.agents.writer_agent import writer_agent
from app.services.storage_service import storage_service

logger = logging.getLogger("app.api.routes")

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    user_id: str

@router.post("/parse")
async def parse_documents(files: List[UploadFile] = File(...), user_id: str = Form("default")):
    """
    **Document Ingestion & RAG Indexing**
    
    Parses multiple documents, extracts structured data/text, and automatically 
    indexes them into a user-specific FAISS vector store.
    
    Supported Formats:
    - Documents: PDF, DOCX, Text
    - Data: CSV, Excel (XLSX, XLS)
    - Multimedia: Images (OCR), Video (AI Parsing)
    - Archives: ZIP (Recursive Processing)
    
    Returns: Structured extraction data + Document ID + (Optional) Data Analysis.
    """
    logger.info(f"Received {len(files)} files for parsing from user: {user_id}")
    results = await route_file(files, user_id=user_id)
    logger.info(f"Parsing complete for {len(results)} files.")
    return {
        "status": "success",
        "user_id": user_id,
        "files_processed": len(results),
        "data": results
    }

@router.post("/ask-doc")
async def ask_document(request: QuestionRequest):
    """
    **Context-Aware Q&A (RAG)**
    
    Uses Retrieval-Augmented Generation to answer questions based on the 
    user's uploaded documents. Employs user-level isolation to ensure 
    data privacy and relevance.
    """
    logger.info(f"Q&A Request from {request.user_id}: {request.question}")
    # Using the rag_agent directly for queries as it's the more modern component
    answer = rag_agent.query(request.question, request.user_id)
    return {"status": "success", "answer": answer}

@router.post("/analyze-data")
async def analyze_data(doc_id: str = Form(...), user_id: str = Form(...), question: str = Form(...)):
    """
    **Deep Data Analysis**
    
    Specialized agentic analysis for tabular data (CSV/Excel). 
    Fetches the original file from storage and uses the Analyst Agent 
    to provide insights, calculate metrics, or answer specific data questions.
    """
    logger.info(f"Analysis Request for Doc ID {doc_id} (User: {user_id}): {question}")
    file_bytes, file_type = storage_service.get_file(doc_id, user_id)
    
    if not file_bytes:
        logger.warning(f"Document {doc_id} not found.")
        return {"status": "error", "message": f"Document ID {doc_id} not found for user {user_id}."}
    
    if file_type in ["xlsx", "xls"]:
        file_type = "excel"

    if file_type not in ["csv", "excel"]:
        logger.warning(f"Unsupported file type for analysis: {file_type}")
        return {"status": "error", "message": f"Only CSV and Excel documents can be analyzed with this tool. Got: {file_type}"}
    
    result = analyst_agent.analyze(file_bytes, file_type, question)
    return result

@router.post("/generate-report")
async def generate_report(request: QuestionRequest):
    """
    **Automated Report Writing**
    
    Retrieves relevant context from the RAG store and passes it to the 
    Writer Agent to generate professional responses:
    - Formal Emails
    - Executive Summaries
    - Technical Reports
    """
    logger.info(f"Report Generation Request from {request.user_id}: {request.question}")

    # Get context from RAG
    context = rag_agent.query(request.question, request.user_id)
    
    # Format state for the writer agent
    writer_state = {
        "query": request.question,
        "user_id": request.user_id,
        "answer": context
    }
    
    # This calls the WriterAgent.__call__ which returns the updated state
    response = writer_agent(writer_state)
    logger.info(f"Report generation completed for user: {request.user_id}")
    return {"status": "success", "data": response.get("written_output")}

@router.post("/clear-index")
async def clear_index():
    """
    **Index Maintenance**
    
    Wipes the FAISS vector store index on disk and clears it from memory.
    """
    # Both services use 'faiss_index' folder, so clearing it from one works for both
    result = await rag_service.clear_index()
    return {"status": "success", "message": result}
