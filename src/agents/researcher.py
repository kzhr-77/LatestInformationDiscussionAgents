import re
import os
from typing import Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults

class ResearcherAgent:
    """
    リサーチャーエージェント
    
    ニュース記事を取得するエージェント。URLまたはキーワードから記事テキストを取得します。
    """
    
    def __init__(self, model: BaseChatModel):
        """
        リサーチャーエージェントを初期化
        
        Args:
            model: LLMモデル（現在は未使用だが、将来的な拡張のため保持）
        """
        self.model = model
        self.tavily_tool = None
        self._init_tavily()
    
    def _init_tavily(self):
        """Tavily検索ツールを初期化（APIキーがある場合のみ）"""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                self.tavily_tool = TavilySearchResults(max_results=3, api_key=api_key)
        except Exception as e:
            print(f"Tavily初期化エラー（キーワード検索は使用できません）: {e}")
    
    def _is_url(self, text: str) -> bool:
        """
        入力がURLかどうかを判定
        
        Args:
            text: 判定するテキスト
        
        Returns:
            URLの場合True
        """
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _fetch_from_url(self, url: str) -> str:
        """
        URLから記事テキストを取得
        
        Args:
            url: 記事のURL
        
        Returns:
            記事のテキスト
        
        Raises:
            ValueError: URLから記事を取得できない場合
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 記事本文を抽出（一般的なHTMLタグから）
            # まず、articleタグを探す
            article = soup.find('article')
            if article:
                text = article.get_text(separator='\n', strip=True)
            else:
                # articleタグがない場合、mainタグを探す
                main = soup.find('main')
                if main:
                    text = main.get_text(separator='\n', strip=True)
                else:
                    # mainタグもない場合、body全体から不要な要素を除外
                    # script, style, nav, footer, headerを削除
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
            
            # テキストを整形（空行を削除、長すぎる行を分割）
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if len(text) < 100:
                raise ValueError("記事テキストが短すぎます。正しいURLか確認してください。")
            
            return text
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"URLから記事を取得できませんでした: {e}")
        except Exception as e:
            raise ValueError(f"記事の解析中にエラーが発生しました: {e}")
    
    def _search_with_tavily(self, query: str) -> str:
        """
        Tavily検索APIを使用して記事を検索
        
        Args:
            query: 検索キーワード
        
        Returns:
            検索結果から取得した記事テキスト
        
        Raises:
            ValueError: Tavilyが利用できない、または検索結果がない場合
        """
        if not self.tavily_tool:
            raise ValueError(
                "Tavily APIキーが設定されていません。\n"
                "環境変数 TAVILY_API_KEY を設定するか、URLを直接入力してください。"
            )
        
        try:
            results = self.tavily_tool.invoke({"query": query})
            
            if not results or len(results) == 0:
                raise ValueError(f"検索キーワード '{query}' に対する結果が見つかりませんでした。")
            
            # 最初の検索結果のURLから記事を取得
            first_result = results[0]
            url = first_result.get('url') if isinstance(first_result, dict) else None
            
            if not url:
                # URLがない場合、contentフィールドを使用
                content = first_result.get('content') if isinstance(first_result, dict) else str(first_result)
                if content:
                    return content
                raise ValueError("検索結果に記事内容が見つかりませんでした。")
            
            # URLから記事を取得
            return self._fetch_from_url(url)
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Tavily検索中にエラーが発生しました: {e}")
    
    def run(self, topic: str) -> str:
        """
        記事を取得するメイン処理
        
        Args:
            topic: 検索キーワードまたはURL
        
        Returns:
            記事のテキスト
        
        Raises:
            ValueError: 記事を取得できない場合
        """
        if not topic or not topic.strip():
            raise ValueError("トピックが指定されていません。")
        
        topic = topic.strip()
        
        # URLかキーワードかを判定
        if self._is_url(topic):
            # URLの場合: 直接コンテンツを取得
            return self._fetch_from_url(topic)
        else:
            # キーワードの場合: Tavily検索APIを使用
            return self._search_with_tavily(topic)

