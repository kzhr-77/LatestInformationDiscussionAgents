# OpenAI用（コメントアウト）
# import os
# from langchain_openai import ChatOpenAI

# Ollama用
from langchain_ollama import ChatOllama
import requests

def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Ollamaサービスへの接続を確認する
    
    Args:
        base_url: OllamaのベースURL
    
    Returns:
        接続可能な場合True
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_llm(model_name: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
    """
    Ollamaを使用してLLMを取得する
    
    Args:
        model_name: 使用するOllamaモデル名（デフォルト: gemma3:4b）
        base_url: OllamaのベースURL（デフォルト: http://localhost:11434）
    
    Returns:
        ChatOllamaインスタンス
    
    Raises:
        ConnectionError: Ollamaサービスに接続できない場合
        ValueError: モデルが存在しない場合
    """
    # 接続確認
    if not check_ollama_connection(base_url):
        raise ConnectionError(
            f"Ollamaサービスに接続できません。\n"
            f"以下の点を確認してください:\n"
            f"1. Ollamaが起動しているか: `ollama serve` または Ollamaアプリが起動しているか\n"
            f"2. URLが正しいか: {base_url}\n"
            f"3. ファイアウォールがブロックしていないか"
        )
    
    # モデルの存在確認
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            if model_name not in models:
                raise ValueError(
                    f"モデル '{model_name}' が見つかりません。\n"
                    f"利用可能なモデル: {', '.join(models)}\n"
                    f"モデルをダウンロードするには: `ollama pull {model_name}`"
                )
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama APIへの接続に失敗しました: {e}")
    
    return ChatOllama(
        model=model_name,
        temperature=0.7,
        base_url=base_url
    )
    
    # OpenAI用（コメントアウト）
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables")
    # return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)

