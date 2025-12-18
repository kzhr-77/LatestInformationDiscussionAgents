# OpenAI用（コメントアウト）
# import os
# from langchain_openai import ChatOpenAI

# Ollama用
from langchain_ollama import ChatOllama
import requests
import json

def _fetch_ollama_tags(base_url: str = "http://localhost:11434") -> dict:
    """
    Ollamaの /api/tags を取得する（モデル一覧取得）。

    Raises:
        ConnectionError: 接続不可/HTTPエラー
        ValueError: JSONとして解釈できない
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            "Ollamaサービスに接続できません。Ollamaが起動していることを確認してください。\n"
            "💡 対処方法: `ollama serve` を実行するか、Ollamaアプリを起動してください。\n"
            f"詳細: {e}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Ollama APIの応答がJSONではありません: {e}")

def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Ollamaサービスへの接続を確認する
    
    Args:
        base_url: OllamaのベースURL
    
    Returns:
        接続可能な場合True
    """
    try:
        _fetch_ollama_tags(base_url)
        return True
    except Exception:
        return False

def get_llm(
    model_name: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    num_predict: int | None = None,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    stop: list[str] | None = None,
    verify_model: bool = True,
):
    """
    Ollamaを使用してLLMを取得する
    
    Args:
        model_name: 使用するOllamaモデル名（デフォルト: gemma3:4b）
        base_url: OllamaのベースURL（デフォルト: http://localhost:11434）
        temperature: 温度パラメータ（デフォルト: 0.7）
        num_predict: 生成する最大トークン数（Ollama側の上限）
        repeat_penalty: 反復抑制（1.0より大きいほど反復しにくい）
        repeat_last_n: 直近Nトークンを反復判定に使う
        stop: 生成停止シーケンス
    
    Returns:
        ChatOllamaインスタンス
    
    Raises:
        ConnectionError: Ollamaサービスに接続できない場合
        ValueError: モデルが存在しない場合
    """
    if verify_model:
        # /api/tags を1回だけ取得して、接続確認とモデル存在確認をまとめて行う
        tags = _fetch_ollama_tags(base_url)
        models = [m.get("name") for m in tags.get("models", []) if isinstance(m, dict) and m.get("name")]
        if model_name not in models:
            raise ValueError(
                f"モデル '{model_name}' が見つかりません。\n"
                f"利用可能なモデル: {', '.join(models) if models else 'なし'}\n"
                f"モデルをダウンロードするには: `ollama pull {model_name}`"
            )
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        num_predict=num_predict,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        stop=stop,
    )
    
    # OpenAI用（コメントアウト）
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables")
    # return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)

