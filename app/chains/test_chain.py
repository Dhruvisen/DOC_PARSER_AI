from app.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_test_chain(text: str):
    """
    A simple test chain that takes text input and returns LLM output.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})
