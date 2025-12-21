from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os
import re
import xml.etree.ElementTree as ET

import requests
from src.utils.security import fetch_url_bytes, validate_outbound_url, UrlValidationError


@dataclass(frozen=True)
class FeedItem:
    title: str
    link: str
    summary: str = ""
    published: str = ""


def load_rss_feed_urls(
    env_var: str = "RSS_FEED_URLS",
    file_path: str = "config/rss_feeds.txt",
) -> list[str]:
    """
    RSS/AtomフィードURLの許可リストを読み込む。
    優先順位: 環境変数 -> 設定ファイル
    """
    # 本番では環境変数注入を避けるため、ファイルのみ使用する設定を推奨（security_spec.md）
    file_only = (os.getenv("RSS_FEEDS_FILE_ONLY") or "").strip() in ("1", "true", "True", "yes", "on")
    if file_only:
        env_val = ""
    else:
        env_val = (os.getenv(env_var) or "").strip()
    if env_val:
        # カンマ区切り or 改行/空白区切りを許容
        parts = re.split(r"[\s,]+", env_val)
        urls = [p.strip() for p in parts if p.strip()]
        return _dedupe_preserve_order(urls)

    path = Path(file_path)
    if path.exists():
        lines: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return _dedupe_preserve_order(lines)

    return []


def fetch_feed_xml(url: str, timeout: int = 10) -> str:
    # security_spec.md: RSS取得もURL検証・サイズ上限・リダイレクト制御を適用する
    _ = validate_outbound_url(url, purpose="rss")
    result = fetch_url_bytes(url, purpose="rss")
    try:
        return result.content.decode("utf-8")
    except Exception:
        # 最低限のフォールバック（XMLはUTF-8以外もあり得る）
        return result.content.decode("utf-8", errors="ignore")


def parse_feed(xml_text: str) -> list[FeedItem]:
    """
    RSS2.0 / Atom の最低限パース（タイトル・リンク・概要を抽出）。
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # namespace除去して扱いやすくする
    _strip_namespaces(root)
    tag = (root.tag or "").lower()

    items: list[FeedItem] = []

    # RSS: <rss><channel><item>...
    if tag == "rss":
        channel = root.find("channel")
        if channel is None:
            return []
        for it in channel.findall("item"):
            title = _text(it.find("title"))
            link = _text(it.find("link"))
            summary = _text(it.find("description"))
            published = _text(it.find("pubDate"))
            if link:
                items.append(FeedItem(title=title, link=link, summary=summary, published=published))
        return items

    # Atom: <feed><entry>...
    if tag == "feed":
        for entry in root.findall("entry"):
            title = _text(entry.find("title"))
            summary = _text(entry.find("summary")) or _text(entry.find("content"))
            published = _text(entry.find("updated")) or _text(entry.find("published"))
            link = ""
            # <link href="..."/> or <link>...</link>
            for l in entry.findall("link"):
                href = (l.attrib or {}).get("href")
                if href:
                    link = href
                    break
                text_link = _text(l)
                if text_link:
                    link = text_link
                    break
            if link:
                items.append(FeedItem(title=title, link=link, summary=summary, published=published))
        return items

    return []


def rank_items_by_query(items: Iterable[FeedItem], query: str, limit: int = 5) -> list[FeedItem]:
    """
    タイトル+概要に対して簡易キーワードマッチし、スコア順に上位を返す。
    - 日本語は空白区切りが効きにくいので、クエリが単語1つの場合は簡易N-gramを併用
    - スコアは「一致トークンの重み（長いほど重い）」の合計
    """
    q = (query or "").strip()
    if not q:
        return []
    tokens = _tokenize_query(q)
    if not tokens:
        return []

    scored: list[tuple[int, FeedItem]] = []
    for it in items:
        hay = f"{it.title}\n{it.summary}".lower()
        hit = 0
        for t in set(tokens):
            tl = t.lower()
            if tl and tl in hay:
                # 短いトークンはノイズになりやすいので重みを抑える
                hit += max(1, min(len(tl), 6))
        if hit > 0:
            scored.append((hit, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [it for _, it in scored[: max(0, limit)]]
    return top


def _tokenize_query(q: str) -> list[str]:
    """
    クエリのトークン化。
    - 空白区切りを基本
    - 日本語等（CJKを含む）で単語1つの場合は2-gramを補助的に追加
    """
    parts = [p for p in re.split(r"\s+", q.strip()) if p]
    if not parts:
        return []

    # すでに複数語なら、そのまま（"AI 技術" など）
    if len(parts) >= 2:
        return parts

    token = parts[0]
    if _has_cjk(token) and len(token) >= 4:
        ngrams = _bigrams(token, max_ngrams=30)
        return [token] + ngrams

    return parts


def _has_cjk(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        # CJK Unified Ideographs / Hiragana / Katakana / CJK Symbols
        if (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0x3000 <= o <= 0x303F):
            return True
    return False


def _bigrams(text: str, max_ngrams: int = 30) -> list[str]:
    out: list[str] = []
    for i in range(max(0, len(text) - 1)):
        out.append(text[i : i + 2])
        if len(out) >= max_ngrams:
            break
    return out


def _strip_namespaces(elem: ET.Element) -> None:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


