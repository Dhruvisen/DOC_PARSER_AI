from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.message import ChatRequest, ContentItem
import os
from dotenv import load_dotenv

load_dotenv()

def run_test_chain(request: ChatRequest):
    """
    Execute a chat completion using the provided messages.
    Supports text and multimodal inputs (images).
    """
    
    # Initialize LLM
    # Use the model from the request, or default to a capable model
    model_name = request.model or "llama-3.3-70b-versatile"
    
    llm = ChatGroq(
        model=model_name,
        temperature=request.temperature,
        max_retries=2,
        # api_key is loaded from env automatically by ChatGroq if GROQ_API_KEY is set
    )

    langchain_messages = []

    for msg in request.messages:
        content = []
        
        if isinstance(msg.content, str):
            # Simple text content
            content = msg.content
        else:
            # List of content items (multimodal)
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": item.text})
                elif item.type == "image_url" and item.image_url:
                    content.append({
                        "type": "image_url", 
                        "image_url": item.image_url
                    })
                # Note: LangChain's support for video_url varies by provider/version.
                # For Groq/Llama Vision, usually images are supported. 
                # If a video is passed, we might need to extract frames first (handled elsewhere).
        
        if msg.role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif msg.role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    # Invoke the chain
    response = llm.invoke(langchain_messages)
    
    return response.content
