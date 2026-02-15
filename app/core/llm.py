from langchain_groq import ChatGroq
from app.core.config import get_settings

def get_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0):
    """
    Returns a LangChain ChatGroq model instance.
    """
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature
    )
