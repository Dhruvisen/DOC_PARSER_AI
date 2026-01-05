from fastapi import FastAPI, UploadFile, File
from app.router import route_file

app = FastAPI(title="AI Document Parser")

@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    result = await route_file(file)
    return {
        "status": "success",
        "filename": file.filename,
        "data": result
    }   