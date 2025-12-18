import os
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults
from src.utils.rss import fetch_feed_xml, load_rss_feed_urls, parse_feed, rank_items_by_query
import logging


class RssKeywordNotFoundError(ValueError):
    """RSSフィード内に検索キーワードの一致が見つからなかった場合の例外。"""


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
        self.rss_feed_urls = load_rss_feed_urls()
    
    def _init_tavily(self):
        """Tavily検索ツールを初期化（APIキーがある場合のみ）"""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                self.tavily_tool = TavilySearchResults(max_results=3, api_key=api_key)
        except Exception as e:
            logging.getLogger(__name__).exception("Tavily初期化エラー（キーワード検索は使用できません）: %s", e)

    def _search_with_rss(self, query: str) -> str:
        """
        RSS/公式フィード許可リストからキーワードに合致する記事URLを探し、本文を取得する。

        設計方針:
        - 許可リスト（環境変数 RSS_FEED_URLS または config/rss_feeds.txt）に限定
        - 無差別クロールはしない
        """
        if not query or not query.strip():
            raise ValueError("検索キーワードが空です。")

        feed_urls = self.rss_feed_urls or load_rss_feed_urls()
        if not feed_urls:
            raise ValueError(
                "RSSフィード許可リストが未設定です。\n"
                "環境変数 RSS_FEED_URLS を設定するか、config/rss_feeds.txt にRSS/AtomのURLを記載してください。"
            )

        # フィードを集約して候補記事を収集
        all_items = []
        for feed_url in feed_urls[:50]:  # 念のため上限
            try:
                xml = fetch_feed_xml(feed_url, timeout=10)
                items = parse_feed(xml)
                all_items.extend(items)
            except Exception as e:
                logging.getLogger(__name__).warning("RSS取得失敗: %s (%s)", feed_url, e)
                continue

        if not all_items:
            raise ValueError("RSSフィードから記事候補を取得できませんでした。")

        ranked = rank_items_by_query(all_items, query=query, limit=5)
        if not ranked:
            raise RssKeywordNotFoundError(f"RSSフィード内にキーワード '{query}' の一致が見つかりませんでした。")

        # 既定は最上位1件（複数記事の混在で分析がブレやすいため）。必要なら環境変数で増やす。
        try:
            max_articles = int(os.getenv("RSS_MAX_ARTICLES", "1"))
        except Exception:
            max_articles = 1
        max_articles = max(1, min(max_articles, 3))

        # 上位から本文を取得（同一URLは除外）
        texts = []
        seen_urls = set()
        for it in ranked:
            url = (it.link or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                # RSS経由は上で[source]/[title]を付与するので、二重ヘッダを避ける
                article = self._fetch_from_url(url, include_header=False)
                header = f"[source] {url}\n[title] {it.title}".strip()
                texts.append(header + "\n\n" + article)
            except Exception as e:
                logging.getLogger(__name__).warning("本文取得失敗: %s (%s)", url, e)
                continue
            if len(texts) >= max_articles:
                break

        if not texts:
            raise ValueError("候補URLから本文を取得できませんでした。")

        if len(texts) == 1:
            return texts[0]
        return "\n\n" + ("\n\n" + ("-" * 40) + "\n\n").join(texts)
    
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
    
    def _fetch_from_url(self, url: str, include_header: bool = True) -> str:
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

            raw_html = response.text

            # 可能なら readability で本文抽出（別記事一覧/ナビ混入を抑える）
            extracted_html = None
            extracted_title = ""
            try:
                from readability import Document  # readability-lxml

                doc = Document(raw_html)
                extracted_html = doc.summary(html_partial=True)
                extracted_title = (doc.short_title() or "").strip()
            except Exception:
                extracted_html = None
                extracted_title = ""

            soup = BeautifulSoup(extracted_html or raw_html, 'lxml')

            # タイトル抽出（後段のレポートで利用）
            def extract_title() -> str:
                if extracted_title:
                    return extracted_title
                # 1) og:title / twitter:title / meta name=title
                try:
                    for sel in [
                        ("meta", {"property": "og:title"}),
                        ("meta", {"name": "twitter:title"}),
                        ("meta", {"name": "title"}),
                    ]:
                        tag = soup.find(sel[0], attrs=sel[1])
                        if tag and tag.get("content"):
                            t = str(tag.get("content")).strip()
                            if t:
                                return t
                except Exception:
                    pass

                # 2) h1（article→main→body優先）
                try:
                    for container in [soup.find("article"), soup.find("main"), soup.find("body"), soup]:
                        if not container:
                            continue
                        h1 = container.find("h1")
                        if h1:
                            t = h1.get_text(separator=" ", strip=True)
                            if t:
                                return t
                except Exception:
                    pass

                # 3) <title>
                try:
                    if soup.title and soup.title.string:
                        t = str(soup.title.string).strip()
                        if t:
                            return t
                except Exception:
                    pass

                return ""

            # まず不要要素を削除（ノイズ混入を減らす）
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'svg']):
                tag.decompose()
            
            # 記事本文を抽出（一般的なHTMLタグから）
            # - サイトによっては <article> が「見出しのみ」で本文が別DOMにあるケースがあるため
            #   短すぎる場合は別の抽出方法へフォールバックする
            text = ""

            def extract_from(container) -> str:
                # 段落中心に拾う（body全文のメニュー等を避ける）
                parts = []
                # li は「関連記事/一覧」を拾いやすいので除外（本文混入対策）
                for el in container.find_all(["h1", "h2", "h3", "p"]):
                    t = el.get_text(separator=" ", strip=True)
                    if not t:
                        continue
                    # 短すぎる断片は捨てる（シェア/ボタン等が混じりやすい）
                    if len(t) < 5:
                        continue
                    parts.append(t)
                return "\n".join(parts)

            # 1) article
            article = soup.find('article')
            if article:
                text = extract_from(article) or article.get_text(separator='\n', strip=True)

            # 2) main
            if len(text) < 200:
                main = soup.find('main')
                if main:
                    text = extract_from(main) or main.get_text(separator='\n', strip=True)

            # 3) body全体（最終フォールバック）
            if len(text) < 200:
                body = soup.find('body') or soup
                text = extract_from(body) or body.get_text(separator='\n', strip=True)
            
            # テキストを整形（空行を削除、長すぎる行を分割）
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # 重複行を除去（ナビ/パンくず等の反復ノイズを軽減）
            deduped = []
            seen = set()
            for line in lines:
                if line in seen:
                    continue
                seen.add(line)
                deduped.append(line)
            lines = deduped
            text = '\n'.join(lines)
            
            if len(text) < 100:
                raise ValueError("記事テキストが短すぎます。正しいURLか確認してください。")

            if include_header:
                title = extract_title()
                header_parts = [f"[source] {url}"]
                if title:
                    header_parts.append(f"[title] {title}")
                header = "\n".join(header_parts).strip()
                return header + "\n\n" + text

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
            return self._fetch_from_url(url, include_header=True)
            
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
            return self._fetch_from_url(topic, include_header=True)
        else:
            # キーワードの場合: RSS許可リスト方式（安全重視）を優先
            try:
                return self._search_with_rss(topic)
            except Exception as rss_err:
                # RSS未設定などの場合のみ、Tavilyが使えるならフォールバック（任意）
                if self.tavily_tool:
                    return self._search_with_tavily(topic)
                raise rss_err

