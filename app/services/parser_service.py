from fastapi import UploadFile, HTTPException
from typing import List, Union
import filetype

from agent.text_parser_agent import TextParserAgent
from app.agents.rag_agent import rag_agent
from loaders.pdf_loader import load_pdf
from loaders.image_loader import ocr_image_from_bytes
from loaders.doc_loader import load_doc
from loaders.csv_loader import load_csv
from loaders.excel_loader import load_excel
from loaders.zip_loader import process_zip_data
from loaders.video_loader import video_parser
from app.services.storage_service import storage_service

agent = TextParserAgent()

async def process_single_file_content(file_bytes: bytes, filename: str, user_id: str = "default"):
    """
    Multimodal Parsing Engine
    -------------------------
    1.  MIME Detection: Identifies file type using magic bytes.
    2.  Routing: Dispatches to specialized loaders (PDF, DOCX, CSV, Excel, Image, Zip, Video).
    3.  Text Extraction: Converts complex formats into clean, agent-ready text.
    4.  Refinement: Uses TextParserAgent (LLM) to structure and clean the raw extraction.
    5.  Deep Data Analysis: If CSV/Excel, uses AnalystAgent for initial insights.
    6.  Storage: Persists the original file.
    7.  RAG Ingestion: Indexes the refined text into the user-partitioned FAISS vector store.
    """
    # ---------- MIME & EXTENSION DETECTION ----------
    # Prioritize extensions for common data formats because 'filetype' often misidentifies Excel as ZIP
    raw_text = None
    pages = None
    file_type = None
    mime = None

    if filename.lower().endswith(".csv"):
        mime = "text/csv"
    elif filename.lower().endswith((".xlsx", ".xls", ".ods")):
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        # Fallback to magic bytes for other types (PDF, Images, etc.)
        kind = filetype.guess(file_bytes)
        mime = kind.mime if kind else None

    # ---------- ROUTING ----------
    if mime == "application/pdf":
        raw_text, pages = load_pdf(file_bytes)
        file_type = "pdf"

    elif mime in {"image/png", "image/jpeg"}:
        raw_text = await ocr_image_from_bytes(file_bytes, agent.llm)
        if raw_text:
            raw_text = raw_text.replace("\x00", "")
        file_type = "image"

    elif mime in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }:
        raw_text = load_doc(file_bytes)
        file_type = "docx"

    elif mime == "text/csv":
        raw_text = load_csv(file_bytes, markdown=True)
        file_type = "csv"

    elif mime in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }:
        raw_text = load_excel(file_bytes, file_name=filename, markdown=True)
        file_type = "excel"

    elif mime == "application/zip":
        # Recursive zip handling
        raw_text = await process_zip_data(
            zip_data=file_bytes,
            llm=agent.llm,
            process_file_fn=lambda file_data, llm, file_path: process_single_file_content(file_data, str(file_path), user_id=user_id)
        )
        file_type = "zip"

    elif mime and mime.startswith("video/"):
        raw_text = await video_parser(file_bytes, agent.llm)
        file_type = "video"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {mime or 'unknown'}"
        )

    # ---------- SAFETY CHECK ----------
    if not raw_text or not isinstance(raw_text, str):
        if file_type == "zip" and raw_text == "":
            raw_text = "Empty or un-processable ZIP file."
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract text from {filename}"
            )

    # ---------- AGENT CALL ----------
    parsed = agent.parse(
        raw_text=raw_text,
        filename=filename,
        file_type=file_type,
        pages=pages,
    )

    # ---------- DATA ANALYSIS (CSV/EXCEL ONLY) ----------
    analysis_insights = None
    if file_type in ["csv", "excel"]:
        from app.agents.analyst_agent import analyst_agent
        # Give it a generic question for initial parsing
        analysis_result = analyst_agent.analyze(
            file_bytes=file_bytes,
            file_type=file_type,
            question="Provide a general high-level summary and identify any interesting trends or outliers in this dataset."
        )
        if analysis_result["status"] == "success":
            analysis_insights = analysis_result["insight"]

    # ---------- STORAGE (MinIO) ----------
    doc_id = storage_service.save_file(file_bytes, filename, user_id)

    # Convert ParsedDocument to dict and add insights/doc_id
    result_data = parsed.model_dump()
    result_data["doc_id"] = doc_id
    if analysis_insights:
        result_data["data_analysis"] = analysis_insights

    # ---------- RAG INGESTION ----------
    await rag_agent.ingest_document(
        text=raw_text,
        filename=filename,
        file_type=file_type,
        user_id=user_id
    )

    return result_data


import asyncio

async def route_file(files: Union[UploadFile, List[UploadFile]], user_id: str = "default"):
    """
    Handles single or multiple file uploads in parallel.
    Returns parsed output per file.
    """
    if not isinstance(files, list):
        files = [files]

    async def _process(file):
        try:
            file_bytes = await file.read()
            filename = file.filename
            parsed = await process_single_file_content(file_bytes, filename, user_id=user_id)
            return {
                "filename": filename,
                "status": "success",
                "data": parsed,
            }
        except HTTPException as e:
            return {
                "filename": file.filename,
                "status": "error",
                "error": e.detail,
            }
        except Exception as e:
            return {
                "filename": file.filename,
                "status": "error",
                "error": str(e),
            }

    # Execute all uploads concurrently
    results = await asyncio.gather(*[_process(f) for f in files])
    return results
