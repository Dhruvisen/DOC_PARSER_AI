from fastapi import APIRouter, UploadFile, File
from app.services.parser_service import route_file
from app.chains.test_chain import run_test_chain

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

@router.get("/test-llm")
async def test_llm(prompt: str = "Hello, how are you?"):
    """
    Test the LLM connection and chat response.
    """
    try:
        response = run_test_chain(prompt)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
