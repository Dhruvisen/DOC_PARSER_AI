from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.message import ChatRequest
from app.core.llm import get_llm

def run_test_chain(request: ChatRequest):
    """
    Execute a chat completion using the provided messages.
    Supports text and multimodal inputs (images).
    """
    
    # Initialize LLM using our unified loader
    # This will respect LLM_PROVIDER and DEFAULT_MODEL from .env
    llm = get_llm(
        model_name=request.model,
        temperature=request.temperature
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
