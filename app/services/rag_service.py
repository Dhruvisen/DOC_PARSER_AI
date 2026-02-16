from typing import List, Optional
import os
import shutil
from pathlib import Path
import uuid

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

    async def ingest_documents(self, files: List[UploadFile], user_id: str) -> str:
        """
        Processes multiple uploaded documents, chunks them, and adds embeddings to the vector store.
        Each chunk is tagged with the user_id for multi-tenant support.
        """
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        all_chunks = []
        processed_files = []
        
        try:
            for file in files:
                file_path = temp_dir / file.filename
                try:
                    doc_id = str(uuid.uuid4())
                    
                    # Save uploaded file temporarily
                    with file_path.open("wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # Load document based on extension
                    if file.filename.endswith(".pdf"):
                        loader = PyMuPDFLoader(str(file_path))
                    elif file.filename.endswith(".txt"):
                        loader = TextLoader(str(file_path), encoding="utf-8")
                    else:
                        print(f"Skipping unsupported file: {file.filename}")
                        continue
                    
                    documents = loader.load()
                    
                    # Add user_id, doc_id and filename to metadata for each document
                    for doc in documents:
                        doc.metadata["user_id"] = user_id
                        doc.metadata["doc_id"] = doc_id
                        doc.metadata["filename"] = file.filename
                    
                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_documents(documents)
                    all_chunks.extend(chunks)
                    processed_files.append(file.filename)
                finally:
                    # Cleanup temp file for each specifically after processing it
                    if file_path.exists():
                        file_path.unlink()

            if not all_chunks:
                 raise HTTPException(status_code=400, detail="No documents were processed or unsupported file types provided.")

            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.vector_store.add_documents(all_chunks)
            
            # Persist the index
            self.vector_store.save_local(self.index_path)
            
            return f"Successfully processed {len(processed_files)} files for user {user_id}: {', '.join(processed_files)} with {len(all_chunks)} total chunks."
            
        finally:
            # Final cleanup of temp dir if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

    async def query_document(self, query: str, user_id: str) -> str:
        """
        Retrieves relevant context for the query (filtered by user_id) and generates an answer using the LLM.
        """
        if self.vector_store is None:
            raise HTTPException(status_code=400, detail="No documents indexed. Please upload a document first.")
        
        # Retrieve relevant chunks filtered by user_id
        try:
            # FAISS in LangChain supports a 'filter' dict for exact matches
            context_docs = self.vector_store.similarity_search(
                query, 
                k=8, 
                filter={"user_id": user_id}
            )
        except Exception as e:
            print(f"Search with filter failed: {e}. Falling back to post-retrieval filtering.")
            # Fallback for older LangChain/FAISS versions or if filter structure is different
            all_results = self.vector_store.similarity_search(query, k=20)
            context_docs = [doc for doc in all_results if doc.metadata.get("user_id") == user_id][:8]

        if not context_docs:
            return "I couldn't find any relevant information in your uploaded documents. Please ensure you have uploaded the documents first."

        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Construct improved prompt
        template = """You are a helpful assistant that answers questions based on the provided context. 
If the answer is not contained within the context, politely state that you don't have enough information to answer the question.

Context:
{context}

Question: {question}

Detailed Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context_text, "question": query})
        
        return answer

    async def clear_index(self) -> str:
        """
        Clears the entire index.
        """
        self.vector_store = None
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        return "Whole index cleared successfully."

# Global instance
rag_service = RAGService()
