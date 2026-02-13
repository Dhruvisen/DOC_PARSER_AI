# AI Document Parser

A FastAPI-based application for parsing and extracting data from various document formats using AI.

## Setup & Installation

### Prerequisites
- Python 3.11+
- `uv` package manager (https://github.com/astral-sh/uv)

### Step 1: Create Virtual Environment
```bash
uv venv
```

### Step 2: Activate Virtual Environment
```bash
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
uv sync
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_api_key_here
```

## Running the Application

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### Using a Different Port
If port 8000 is already in use, specify a different port:
```bash
uvicorn app.main:app --reload --port 8001
```

### Kill Existing Process on Port 8000
If you get "Address already in use" error:
```bash
lsof -ti:8000 | xargs kill -9
```

## Project Structure

```
├── app/                 # FastAPI application
│   ├── main.py         # Entry point
│   └── router.py       # Route handlers
├── agent/              # AI agent modules
├── llm/                # LLM models
├── loaders/            # Document loaders (PDF, CSV, Excel, etc.)
├── schemas/            # Output schemas
└── utils/              # Utility functions
```

## API Endpoints

### POST /parse
Upload and parse a document.

**Request:**
- `file`: Document file (multipart/form-data)

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "data": { }
}
```

## Supported Document Types
- PDF
- Word Documents (.docx)
- CSV
- Excel (.xlsx)
- Images
- Video
- ZIP archives
- Text files
