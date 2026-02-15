from typing import List, Optional
import os
import shutil
from pathlib import Path

from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings

settings = get_settings()

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2,
            api_key=settings.GROQ_API_KEY
        )
        self.index_path = "faiss_index"

        # Load existing index if available
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Required for loading pickle files safely
                )
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
                self.vector_store = None

    async def ingest_document(self, file: UploadFile) -> str:
        """
        Processes an uploaded document, chunks it, and adds embeddings to the vector store.
        Supports PDF and TXT files.
        """
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / file.filename
        
        try:
            # Save uploaded file temporarily
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Load document based on extension
            if file.filename.endswith(".pdf"):
                loader = PyMuPDFLoader(str(file_path))
            elif file.filename.endswith(".txt"):
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT are supported for RAG.")
            
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                 raise HTTPException(status_code=400, detail="Document appears to be empty or could not be processed.")

            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            
            # Persist the index
            self.vector_store.save_local(self.index_path)
            
            return f"Successfully processed {file.filename} with {len(chunks)} chunks."
            
        finally:
            # Cleanup temp file
            if file_path.exists():
                file_path.unlink()
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

    async def query_document(self, query: str) -> str:
        """
        Retrieves relevant context for the query and generates an answer using the LLM.
        """
        if self.vector_store is None:
            raise HTTPException(status_code=400, detail="No documents indexed. Please upload a document first.")
        
        # Retrieve relevant chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        context_docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Construct prompt
        template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": query})
        
        return answer

# Global instance
rag_service = RAGService()
