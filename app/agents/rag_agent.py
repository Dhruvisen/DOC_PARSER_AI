import os
import logging
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.llm import get_llm
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("app.agents.rag")

class RAGAgent:
    def __init__(self, index_path: str = "faiss_index"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_path = index_path
        self.llm = get_llm(temperature=0)
        self.vector_store = self._load_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        logger.info("RAGAgent initialized.")

    def _load_vector_store(self):
        """Loads the FAISS vector store from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                # Use absolute path or ensure current working directory is correct
                return FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load index at {self.index_path}: {e}")
        return None

    def ingest_document(self, text: str, filename: str, file_type: str, user_id: str = "default"):
        """
        Stores parsed text into the vector database with metadata.
        """
        logger.info(f"Ingesting document: {filename} (Type: {file_type}) for User: {user_id}")
        doc = Document(
            page_content=text,
            metadata={
                "filename": filename,
                "file_type": file_type,
                "user_id": user_id
            }
        )
        chunks = self.text_splitter.split_documents([doc])
        logger.info(f"Split into {len(chunks)} chunks.")
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        
        # Persist data
        self.vector_store.save_local(self.index_path)
        logger.info(f"Successfully indexed and saved {filename}")
        return len(chunks)

    def query(self, query: str, user_id: str = "default") -> str:
        """
        Retriever chain that answers questions based ONLY on stored documents.
        """
        logger.info(f"RAG Query from {user_id}: {query}")
        
        if self.vector_store is None:
            logger.warning("No vector store found during query.")
            return "No documents have been indexed yet. Please upload a document first."

        # Filter by user_id to ensure strict multi-tenancy
        try:
            results = self.vector_store.similarity_search(
                query, 
                k=5, 
                filter={"user_id": user_id}
            )
        except Exception as e:
            # Fallback: Retrieve more and filter in-memory if exact metadata filtering fails
            logger.error(f"Filter search failed: {e}. Falling back to in-memory filtering.")
            all_results = self.vector_store.similarity_search(query, k=20)
            results = [doc for doc in all_results if doc.metadata.get("user_id") == user_id][:5]

        if not results:
            logger.info("No relevant documents found for the query.")
            return "I couldn't find any relevant information in the documents for your request."

        logger.info(f"Found {len(results)} relevant chunks. Generating answer...")
        
        # Build context from relevant chunks
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # Define response prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a sophisticated document analysis assistant. Use the following context to answer the user's question.
        
        CRITICAL GUIDELINES:
        - If the context contains Markdown tables (spreadsheets), interpret them as rows and columns.
        - Answer the question ONLY based on the context provided. 
        - If the context is a spreadsheet and the user asks for a summary, describe the columns and notable data points.
        - If you don't know the answer or the context doesn't contain it, state that you don't have enough information from the documents.
        - Always provide a professional and detailed response.

        Context:
        {context}

        Question: {question}

        Detailed Answer:""")

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": query})
        logger.info("Answer generated successfully.")
        return answer

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callable interface for LangGraph nodes.
        Expects 'query' and 'user_id' in state.
        Stores 'answer' in the state dictionary.
        """
        query = state.get("query")
        user_id = state.get("user_id", "default")
        
        if not query:
            return {**state, "answer": "Error: No query provided in state."}
            
        answer = self.query(query, user_id)
        
        # Update state with the answer
        return {**state, "answer": answer}

# Export instance for easy integration
rag_agent = RAGAgent()
