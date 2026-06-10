# 📄 RAG Doc Engine

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

### 4. **Modular Agentic Design**
Parsing logic is exposed as reusable LangChain tools, ready for integration into larger AI ecosystems.

---

## 🛠️ Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Frontend**: [Streamlit](https://streamlit.io/) (Interactive UI)
- **AI Framework**: [LangChain](https://www.langchain.com/) & [LangGraph](https://python.langchain.com/docs/langgraph)
- **LLM Providers**: 
  - 🌩️ **Groq**: Llama 3 models
  - 💎 **Google Gemini**: Pro/Flash models
  - 🏠 **Ollama**: Local models (Llama 3, Mistral, etc.)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [HuggingFace](https://huggingface.co/) (MiniLM)
- **Storage**: [MinIO](https://min.io/) (S3-Compatible Object Storage)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- `uv` package manager (Recommended)
- [Optional] Ollama for local LLM usage
- [Optional] MinIO for cloud-native storage

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
APP_NAME="rag_doc_engin"
# Provider: groq, google, or ollama
LLM_PROVIDER=groq 

# API Keys
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
OLLAMA_BASE_URL=http://localhost:11434

# Storage (MinIO)
STORAGE_TYPE=minio
MINIO_ENDPOINT=localhost:9022
MINIO_ACCESS_KEY=abc
MINIO_SECRET_KEY=abc_password
```

---

## 🏃 Running the Application

### 0. Start Storage (MinIO)
```bash
docker-compose up -d
```

### 1. Start the Backend (API)
```bash
uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend (UI)
```bash
streamlit run app/frontend/app.py
```

- **Backend API**: `http://localhost:8000`
- **Frontend UI**: `http://localhost:8501`
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

### 🧹 `POST /clear-index`
Clear the vector store index.

---

## 📁 Project Structure

```
├── app/
│   ├── agents/         # AI Agents (RAG, Analyst)
│   ├── api/            # Route handlers and API definitions
│   ├── core/           # Configuration, logging, and security
│   ├── services/       # Business logic (Parser, RAG, Storage)
│   └── models/         # Pydantic models
├── loaders/            # Specialized document type loaders
├── agent/              # Legacy / Refactored agent modules
├── schemas/            # Output data schemas
└── utils/              # Helper utilities
```

