import os
from langchain_openai import ChatOpenAI

def get_llm(model_name: str = "gpt-4o"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)

