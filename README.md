# 📄 AI Document Parser & RAG System

A modular, high-performance AI backend designed for advanced document processing, multi-tenant Retrieval-Augmented Generation (RAG), and agentic workflows. Built with FastAPI, LangChain, and Groq.

---

## 🚀 Core Capabilities

### 1. **Multimodal Document Parsing**
Extract structured data and text from a wide range of formats:
- 📑 **PDF**: High-fidelity text extraction and page-by-page analysis.
- 🖼️ **Images**: OCR-powered text extraction from PNG and JPEG.
- 📝 **Word**: Seamless processing of `.docx` and `.doc` files.
- 📊 **Structured Data**: Deep cleaning and parsing of CSV and Excel (`.xlsx`) files.
- 📦 **ZIP Archives**: Recursive extraction and processing of files within compressed folders.
- 🎥 **Video**: AI-driven analysis of video content.

### 2. **Multi-Tenant RAG System**
- **User Isolation**: Ingest documents into a FAISS vector store with strict user-level partitioning.
- **Efficient Retrieval**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) for fast and accurate semantic search.
- **Intelligent Q&A**: Ask complex questions specifically about your uploaded documents.

### 3. **AI-Driven Analytics**
- **Automated Summarization**: Automatically generates high-level summaries and cleans extracted text.
- **Data Insights**: Specialized analysis for CSV/Excel files to identify trends, outliers, and key metrics using the `AnalystAgent`.

### 4. **Agentic Workflows**
- **Writer Agent**: Generates professional emails, summaries, and bulleted reports based on document context.
- **Modular Design**: Parsing logic is exposed as reusable LangChain tools, ready for integration into larger AI ecosystems.

---

## 🛠️ Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI Framework**: [LangChain](https://www.langchain.com/) & [LangGraph](https://python.langchain.com/docs/langgraph)
- **LLM Provider**: [Groq](https://groq.com/) (Llama 3 models)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [HuggingFace](https://huggingface.co/) (MiniLM)
- **Storage**: MinIO / Local Disk

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- `uv` package manager (Recommended)

### Step 1: Initialize Environment
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Configure Environment
Create a `.env` file in the root directory:
```env
APP_NAME="AI Document Parser"
GROQ_API_KEY=your_api_key_here
STORAGE_TYPE=local  # or minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=abc
MINIO_SECRET_KEY=abcpwd
```

---

## 🏃 Running the Application

```bash
uvicorn app.main:app --reload
```

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`

---

## 🛣️ API Endpoints

### 📤 `POST /parse`
Upload documents for parsing and automatic RAG indexing.
- **Payload**: `files` (Multipart), `user_id` (String)
- **Supports**: PDF, DOCX, CSV, XLSX, Images, Video, ZIP.

### 💬 `POST /ask-doc`
Query your documents using semantic search.
- **Payload**: `{"question": "...", "user_id": "..."}`

### 📉 `POST /analyze-data`
Perform deep analysis on CSV/Excel files.
- **Payload**: `doc_id` (Form), `user_id` (Form), `question` (Form)

### ✍️ `POST /generate-report`
Generate professional summaries and reports.
- **Payload**: `{"question": "Report requirements", "user_id": "..."}`

### 🧹 `POST /clear-index`
Clear the vector store index.

---

## 📁 Project Structure

```
├── app/
│   ├── agents/         # AI Agents (RAG, Analyst, Writer)
│   ├── api/            # Route handlers and API definitions
│   ├── core/           # Configuration, logging, and security
│   ├── services/       # Business logic (Parser, RAG, Storage)
│   └── models/         # Pydantic models
├── loaders/            # Specialized document type loaders
├── agent/              # Legacy / Refactored agent modules
├── schemas/            # Output data schemas
└── utils/              # Helper utilities
```

