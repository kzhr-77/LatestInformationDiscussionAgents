from __future__ import annotations

import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from typing import Iterable, Literal
from urllib.parse import urljoin, urlparse, urlunparse

import requests


class UrlValidationError(ValueError):
    """URLが危険/不正で、外部アクセスを拒否する。"""


class OutboundHttpError(ConnectionError):
    """外部HTTPアクセスに失敗（接続/ステータス/タイムアウト等）。"""


class ResponseTooLargeError(ValueError):
    """レスポンスがサイズ上限を超過。"""


Purpose = Literal["article", "rss"]


@dataclass(frozen=True)
class FetchResult:
    url: str
    content: bytes
    content_type: str


_BLOCKED_V4 = [
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
]

_BLOCKED_V6 = [
    "::/128",
    "::1/128",
    "fe80::/10",
    "fc00::/7",
    "ff00::/8",
]

_BLOCKED_NETS = [ipaddress.ip_network(x) for x in (_BLOCKED_V4 + _BLOCKED_V6)]


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    try:
        return int(v) if v else default
    except Exception:
        return default


def _split_list_env(name: str) -> list[str]:
    v = (os.getenv(name) or "").strip()
    if not v:
        return []
    parts = re.split(r"[\s,]+", v)
    return [p.strip() for p in parts if p.strip()]


def sanitize_url_for_logging(url: str, max_chars: int = 200) -> str:
    """
    ログ用のURL整形（クエリ/フラグメントをマスク、改行除去、長さ制限）。
    """
    s = "" if url is None else str(url)
    s = re.sub(r"[\r\n\t]+", " ", s).strip()
    try:
        p = urlparse(s)
        if p.scheme and p.netloc:
            # クエリ/フラグメントをマスク
            p2 = p._replace(query="…", fragment="")
            s = urlunparse(p2)
    except Exception:
        pass
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def is_blocked_ip(ip: str) -> bool:
    """
    拒否IPレンジ判定（IPv4/IPv6、IPv4-mapped IPv6は展開して判定）。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return True  # 解析不能は安全側で拒否

    # IPv4-mapped IPv6 (::ffff:a.b.c.d) はIPv4へ展開
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    for net in _BLOCKED_NETS:
        if addr in net:
            return True
    return False


def resolve_host_ips(hostname: str) -> list[str]:
    """
    ホスト名を解決してIP一覧を返す（DNSリバインディング対策の材料）。
    """
    if not hostname:
        return []
    out: list[str] = []
    try:
        infos = socket.getaddrinfo(hostname, None)
        for fam, _, _, _, sockaddr in infos:
            if fam == socket.AF_INET:
                out.append(sockaddr[0])
            elif fam == socket.AF_INET6:
                out.append(sockaddr[0])
    except Exception:
        return []
    # 重複除去
    uniq: list[str] = []
    seen: set[str] = set()
    for ip in out:
        if ip in seen:
            continue
        seen.add(ip)
        uniq.append(ip)
    return uniq


def _domain_allowed(host: str, allowlist: Iterable[str]) -> bool:
    """
    ドメイン許可:
    - 完全一致
    - もしくはサブドメイン（host.endswith("." + domain)）
    - allowlist側の "*.example.com" は "example.com" として扱う
    """
    h = (host or "").lower().rstrip(".")
    if not h:
        return False
    for d in allowlist:
        dom = (d or "").strip().lower().rstrip(".")
        if not dom:
            continue
        if dom.startswith("*."):
            dom = dom[2:]
        if h == dom:
            return True
        if h.endswith("." + dom):
            return True
    return False


def validate_outbound_url(url: str, *, purpose: Purpose) -> str:
    """
    外部HTTPアクセス用URL検証（SSRF対策）。
    返り値は正規化済みURL（文字列）。違反時は UrlValidationError。
    """
    raw = "" if url is None else str(url).strip()
    if not raw:
        raise UrlValidationError("URLが空です。")

    p = urlparse(raw)
    if not p.scheme or not p.netloc:
        raise UrlValidationError("URLの形式が不正です。")

    # スキーム
    allowed_schemes = [s.lower() for s in (_split_list_env("URL_ALLOWED_SCHEMES") or ["https"])]
    scheme = (p.scheme or "").lower()
    if scheme not in allowed_schemes:
        raise UrlValidationError(f"許可されていないスキームです: {scheme}")

    # userinfo禁止
    if p.username or p.password:
        raise UrlValidationError("ユーザー名/パスワード付きURLは許可されません。")

    host = (p.hostname or "").strip().lower()
    if not host:
        raise UrlValidationError("ホスト名が不正です。")
    if host in ("localhost", "localhost."):
        raise UrlValidationError("localhost宛先は許可されません。")

    # ドメイン許可リスト（任意）
    allowlist = _split_list_env("URL_ALLOWLIST_DOMAINS")
    if allowlist and not _domain_allowed(host, allowlist):
        raise UrlValidationError("許可リスト外のドメインです。")

    # DNS解決 + 内部IP拒否
    if _env_bool("URL_BLOCK_PRIVATE_IPS", True):
        ips = resolve_host_ips(host)
        if not ips:
            raise UrlValidationError("ホスト名を解決できませんでした。")
        for ip in ips:
            if is_blocked_ip(ip):
                raise UrlValidationError(f"危険な宛先（内部/予約IP）へのアクセスは拒否されました: {ip}")

    # 返すURLは、スキーム/ネットロケーションはそのまま（query/fragmentもそのまま）
    return urlunparse(p)


def fetch_url_bytes(url: str, *, purpose: Purpose, headers: dict | None = None) -> FetchResult:
    """
    検証済みURLに対して外部HTTPアクセスを行い、サイズ上限・リダイレクト制御を適用して bytes を返す。
    """
    # 検証
    current = validate_outbound_url(url, purpose=purpose)

    connect_timeout = float(_env_int("HTTP_CONNECT_TIMEOUT_SEC", 3))
    read_timeout = float(_env_int("HTTP_READ_TIMEOUT_SEC", 7))
    timeout = (connect_timeout, read_timeout)

    allow_redirects = _env_bool("URL_ALLOW_REDIRECTS", False)
    max_redirects = max(0, _env_int("URL_MAX_REDIRECTS", 2))

    max_bytes = _env_int("HTTP_MAX_BYTES", 5_000_000)
    if purpose == "rss":
        max_bytes = _env_int("RSS_MAX_BYTES", 2_000_000)

    # Content-Type 許可（ゆるめ）
    if purpose == "article":
        allowed_ct_prefixes = ("text/html", "application/xhtml", "text/plain")
    else:
        allowed_ct_prefixes = ("application/rss", "application/atom", "application/xml", "text/xml", "text/plain")

    sess = requests.Session()
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    if headers:
        hdrs.update(headers)

    redirects = 0
    while True:
        try:
            res = sess.get(current, headers=hdrs, timeout=timeout, stream=True, allow_redirects=False)
        except requests.exceptions.RequestException as e:
            raise OutboundHttpError(f"外部HTTPアクセスに失敗しました: {e}")

        # リダイレクト
        if res.status_code in (301, 302, 303, 307, 308):
            if not allow_redirects:
                raise OutboundHttpError(f"リダイレクトは禁止されています（{res.status_code}）。")
            if redirects >= max_redirects:
                raise OutboundHttpError("リダイレクト回数上限を超えました。")
            loc = res.headers.get("Location") or ""
            if not loc:
                raise OutboundHttpError("リダイレクト先が不明です。")
            nxt = urljoin(current, loc)
            current = validate_outbound_url(nxt, purpose=purpose)
            redirects += 1
            continue

        try:
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise OutboundHttpError(f"HTTPエラー: {e}")

        ct = (res.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ct and not ct.startswith(allowed_ct_prefixes):
            raise OutboundHttpError(f"想定外のContent-Typeです: {ct}")

        cl = res.headers.get("Content-Length")
        if cl:
            try:
                if int(cl) > max_bytes:
                    raise ResponseTooLargeError("レスポンスがサイズ上限を超えています。")
            except ResponseTooLargeError:
                raise
            except Exception:
                # Content-Length が数値でない場合はストリーム上限で制御する
                pass

        buf = bytearray()
        try:
            for chunk in res.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ResponseTooLargeError("レスポンスがサイズ上限を超えました。")
        finally:
            res.close()

        return FetchResult(url=current, content=bytes(buf), content_type=ct)


