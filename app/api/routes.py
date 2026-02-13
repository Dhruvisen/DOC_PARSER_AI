from fastapi import APIRouter, UploadFile, File
from app.services.parser_service import route_file

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
