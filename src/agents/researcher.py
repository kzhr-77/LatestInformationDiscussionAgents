import os
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults
from src.utils.rss import fetch_feed_xml, load_rss_feed_urls, parse_feed, rank_items_by_query
from src.utils.security import fetch_url_bytes, validate_outbound_url, UrlValidationError
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
                items = parse_feed(xml, feed_url=feed_url)
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
                # security_spec.md: RSS item.link 方針（A案/B案）
                # - A案（安全最優先）: フィードと同一ドメイン、または URL_ALLOWLIST_DOMAINS に含まれる場合のみ取得
                # - B案（柔軟）: 取得可。ただし validate_outbound_url は必須
                policy = (os.getenv("RSS_ITEM_LINK_POLICY") or "A").strip().upper()
                if policy not in ("A", "B"):
                    policy = "A"
                if policy == "A":
                    try:
                        feed_host = (urlparse(getattr(it, "feed_url", "") or "").hostname or "").lower().strip(".")
                        item_host = (urlparse(url).hostname or "").lower().strip(".")
                    except Exception:
                        feed_host = ""
                        item_host = ""
                    allowlist = (os.getenv("URL_ALLOWLIST_DOMAINS") or "").strip()
                    # allowlist は security.py 側で解釈されるので、ここでは「同一ドメイン」だけ先に絞る
                    if feed_host and item_host and item_host != feed_host and not item_host.endswith("." + feed_host):
                        # allowlist による許可は validate_outbound_url で判定される（URL_ALLOWLIST_DOMAINS が設定されていれば通る）
                        # ただし allowlist 未設定の場合はここでスキップする
                        if not allowlist:
                            logging.getLogger(__name__).info("RSS item.link をスキップ（A案: feed外ドメイン）: feed=%s item=%s", feed_host, item_host)
                            continue
                        # allowlist がある場合は validate_outbound_url に任せる（通らなければ例外になる）
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
            # security_spec.md: URL直入力/RSS由来URLともにSSRF対策の検証を必須化
            safe_url = validate_outbound_url(url, purpose="article")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            fetched = fetch_url_bytes(safe_url, purpose="article", headers=headers)
            raw_html = fetched.content.decode("utf-8", errors="ignore")

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

            # readability が短すぎる/空の場合は、生HTMLにフォールバック（サイトによっては本文が落ちる）
            def _html_text_len(html: str | None) -> int:
                if not html:
                    return 0
                try:
                    return len(BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True))
                except Exception:
                    return 0

            if extracted_html and _html_text_len(extracted_html) < 200:
                extracted_html = None

            soup = BeautifulSoup(extracted_html or raw_html, 'lxml')

            # タイトル抽出（後段のレポートで利用）
            def _clean_title(t: str) -> str:
                s = (t or "").strip()
                s = " ".join(s.split())
                if not s:
                    return ""
                # サイト名サフィックスを落としやすい区切りを試す（長さが極端に短くなる場合は採用しない）
                seps = [" | ", " - ", "｜", "–", "—", "：", ":"]
                best = s
                for sep in seps:
                    if sep in s:
                        head = s.split(sep, 1)[0].strip()
                        if 8 <= len(head) <= len(best):
                            best = head
                return best.strip()

            def extract_title() -> str:
                if extracted_title:
                    return _clean_title(extracted_title)
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
                                return _clean_title(t)
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
                                return _clean_title(t)
                except Exception:
                    pass

                # 3) <title>
                try:
                    if soup.title and soup.title.string:
                        t = str(soup.title.string).strip()
                        if t:
                            return _clean_title(t)
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

            def select_best_container(s: BeautifulSoup) -> str:
                """
                article/mainが無い or 本文が落ちるサイト向けの追加ヒューリスティック。
                いくつかの「本文っぽい」コンテナ候補から、段落テキスト量が最大のものを採用する。
                """
                selectors = [
                    "article",
                    "main",
                    "div[role='main']",
                    "[itemprop='articleBody']",
                    "#content",
                    ".content",
                    ".article",
                    ".post",
                    ".entry-content",
                    ".post-content",
                    ".article-body",
                    ".story-body",
                    ".main-content",
                ]
                candidates = []
                try:
                    for sel in selectors:
                        candidates.extend(list(s.select(sel))[:10])
                except Exception:
                    candidates = []
                # 重複除去（idベース）
                uniq = []
                seen_ids = set()
                for el in candidates:
                    try:
                        k = id(el)
                    except Exception:
                        k = None
                    if k is None or k in seen_ids:
                        continue
                    seen_ids.add(k)
                    uniq.append(el)
                best_text = ""
                best_len = 0
                for el in uniq[:50]:
                    try:
                        t = extract_from(el)
                    except Exception:
                        continue
                    # 極端に短いコンテナは無視
                    tl = len(t or "")
                    if tl > best_len:
                        best_len = tl
                        best_text = t
                return best_text

            # 1) article
            article = soup.find('article')
            if article:
                text = extract_from(article) or article.get_text(separator='\n', strip=True)

            # 2) main
            if len(text) < 200:
                main = soup.find('main')
                if main:
                    text = extract_from(main) or main.get_text(separator='\n', strip=True)

            # 2.5) 本文っぽいコンテナの最大選択（サイト別DOM差異の吸収）
            if len(text) < 200:
                picked = select_best_container(soup)
                if picked and len(picked) > len(text):
                    text = picked

            # 3) body全体（最終フォールバック）
            if len(text) < 200:
                body = soup.find('body') or soup
                text = extract_from(body) or body.get_text(separator='\n', strip=True)

            # readability利用時に短文になりやすいサイト向け: 生HTMLで再抽出を試す
            if extracted_html and len(text) < 200:
                try:
                    soup2 = BeautifulSoup(raw_html, "lxml")
                    for tag in soup2(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'svg']):
                        tag.decompose()
                    text2 = ""
                    article2 = soup2.find("article")
                    if article2:
                        text2 = extract_from(article2) or article2.get_text(separator="\n", strip=True)
                    if len(text2) < 200:
                        main2 = soup2.find("main")
                        if main2:
                            text2 = extract_from(main2) or main2.get_text(separator="\n", strip=True)
                    if len(text2) < 200:
                        body2 = soup2.find("body") or soup2
                        text2 = extract_from(body2) or body2.get_text(separator="\n", strip=True)
                    if len(text2) > len(text):
                        soup = soup2
                        text = text2
                except Exception:
                    pass
            
            # テキストを整形（空行を削除、長すぎる行を分割）
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # ナビ/フッタっぽい短文を軽く除外（最終フォールバック由来の混入対策）
            noise_tokens = [
                "ログイン",
                "会員登録",
                "メニュー",
                "ホーム",
                "プライバシー",
                "利用規約",
                "Cookie",
                "©",
                "All rights reserved",
                "シェア",
                "フォロー",
                "人気記事",
                "関連記事",
                "次の記事",
                "前の記事",
            ]
            filtered_lines = []
            for ln in lines:
                if len(ln) <= 3:
                    continue
                if any(tok in ln for tok in noise_tokens) and len(ln) <= 40:
                    continue
                # URLっぽい行は除外
                if "http://" in ln or "https://" in ln:
                    continue
                filtered_lines.append(ln)
            lines = filtered_lines
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
            
            if len(text) < 120:
                raise ValueError("記事テキストが短すぎます。正しいURLか確認してください。")

            if include_header:
                title = extract_title()
                header_parts = [f"[source] {safe_url}"]
                if title:
                    header_parts.append(f"[title] {title}")
                header = "\n".join(header_parts).strip()
                return header + "\n\n" + text

            return text
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"URLから記事を取得できませんでした: {e}")
        except UrlValidationError as e:
            raise ValueError(f"危険/不正なURLのため取得を拒否しました: {e}")
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
            # security_spec.md: URL直入力を運用で無効化できるようにする
            if (os.getenv("ALLOW_URL_FETCH") or "").strip() in ("0", "false", "False", "no", "off"):
                raise ValueError("URL直入力による取得は無効です（ALLOW_URL_FETCH=0）。")
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

