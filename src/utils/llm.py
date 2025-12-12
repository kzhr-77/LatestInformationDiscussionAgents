import os
# OpenAI用（コメントアウト）
# from langchain_openai import ChatOpenAI

# Ollama用
from langchain_ollama import ChatOllama

def get_llm(model_name: str = "gemma3:4b"):
    """
    Ollamaを使用してLLMを取得する
    
    Args:
        model_name: 使用するOllamaモデル名（デフォルト: gemma3:4b）
    
    Returns:
        ChatOllamaインスタンス
    """
    return ChatOllama(
        model=model_name,
        temperature=0.7,
        base_url="http://localhost:11434"  # OllamaのデフォルトURL
    )
    
    # OpenAI用（コメントアウト）
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables")
    # return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)

