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

agent = TextParserAgent()

async def process_single_file_content(file_bytes: bytes, filename: str, user_id: str = "default"):
    """
    Core logic to route file bytes to the correct loader and then to the agent.
    Also stores the text in RAG via rag_agent.
    """
    # ---------- MIME DETECTION (SAFE) ----------
    kind = filetype.guess(file_bytes)
    mime = kind.mime if kind else None

    # Fallback for CSV (detection often fails)
    if mime is None and filename.lower().endswith(".csv"):
        mime = "text/csv"

    raw_text = None
    pages = None
    file_type = None

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
        raw_text = load_csv(file_bytes, markdown=False)
        file_type = "csv"

    elif mime in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }:
        raw_text = load_excel(file_bytes, file_name=filename, markdown=False)
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

    # ---------- RAG INGESTION ----------
    # Store the extracted text into the vector store as requested
    rag_agent.ingest_document(
        text=raw_text,
        filename=filename,
        file_type=file_type,
        user_id=user_id
    )

    # Convert ParsedDocument to dict and add insights
    result_data = parsed.model_dump()
    if analysis_insights:
        result_data["data_analysis"] = analysis_insights

    return result_data


async def route_file(files: Union[UploadFile, List[UploadFile]], user_id: str = "default"):
    """
    Handles single or multiple file uploads safely.
    Returns parsed output per file.
    """

    if not isinstance(files, list):
        files = [files]

    results = []

    for file in files:
        try:
            file_bytes = await file.read()
            filename = file.filename

            parsed = await process_single_file_content(file_bytes, filename, user_id=user_id)

            results.append({
                "filename": filename,
                "status": "success",
                "data": parsed,
            })

        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": e.detail,
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e),
            })

    return results
