from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from app.core.config import get_settings

def get_llm(provider: str = None, model_name: str = None, temperature: float = 0):
    """
    Returns a LangChain Chat model instance based on the provider.
    Supported providers: groq, google, ollama.
    """
    settings = get_settings()
    provider = provider or settings.LLM_PROVIDER
    
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=model_name or settings.DEFAULT_MODEL,
            temperature=temperature
        )
    
    elif provider == "google":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=model_name or settings.DEFAULT_MODEL,
            temperature=temperature
        )
    
    elif provider == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name or settings.DEFAULT_MODEL,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
