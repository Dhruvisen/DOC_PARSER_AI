from fastapi import UploadFile, HTTPException
from typing import List, Union
import filetype

from agent.text_parser_agent import TextParserAgent
from loaders.pdf_loader import load_pdf
from loaders.image_loader import ocr_image_from_bytes
from loaders.doc_loader import load_doc
from loaders.csv_loader import load_csv
from loaders.excel_loader import load_excel

agent = TextParserAgent()


async def route_file(files: Union[UploadFile, List[UploadFile]]):
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

            # ---------- MIME DETECTION (SAFE) ----------
            kind = filetype.guess(file_bytes)
            mime = kind.mime if kind else None

            raw_text = None
            pages = None
            file_type = None

            # ---------- ROUTING ----------
            if mime == "application/pdf":
                raw_text, pages = load_pdf(file_bytes)
                print("DEBUG: PDF raw_text length =", len(raw_text))
                print("load_pdf returned", len(pages), "pages")
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
                raw_text = load_csv(file_bytes)
                file_type = "csv"

            elif mime in {
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }:
                raw_text = load_excel(file_bytes)
                file_type = "excel"

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {mime or 'unknown'}"
                )

            # ---------- SAFETY CHECK ----------
            if not raw_text or not isinstance(raw_text, str):
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
            print("DEBUG: Parsed document summary =", parsed.summary)

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
