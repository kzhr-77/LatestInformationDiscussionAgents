import os
import unittest
from unittest.mock import patch

from src.utils.security import (
    OutboundHttpError,
    ResponseTooLargeError,
    UrlValidationError,
    fetch_url_bytes,
    sanitize_url_for_logging,
    validate_outbound_url,
)


class TestSecurityUtils(unittest.TestCase):
    def test_sanitize_url_masks_query_and_removes_newlines(self):
        s = sanitize_url_for_logging("https://example.com/path?token=SECRET\nx=1#frag")
        self.assertNotIn("\n", s)
        self.assertIn("https://example.com/path", s)
        self.assertIn("?…", s)
        self.assertNotIn("SECRET", s)

    def test_rejects_file_scheme(self):
        with self.assertRaises(UrlValidationError):
            validate_outbound_url("file:///etc/passwd", purpose="article")

    def test_rejects_userinfo(self):
        with self.assertRaises(UrlValidationError):
            validate_outbound_url("https://user:pass@example.com/", purpose="article")

    def test_rejects_localhost(self):
        with self.assertRaises(UrlValidationError):
            validate_outbound_url("https://localhost:11434/", purpose="article")

    def test_rejects_private_ip_via_dns_resolution(self):
        # example.com が 127.0.0.1 を返す想定
        def fake_getaddrinfo(host, *args, **kwargs):
            return [(2, None, None, None, ("127.0.0.1", 0))]

        with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with self.assertRaises(UrlValidationError):
                validate_outbound_url("https://example.com/news", purpose="article")

    def test_allowlist_blocks_nonlisted_domain(self):
        # allowlistを有効化し、許可外は拒否される
        with patch.dict(os.environ, {"URL_ALLOWLIST_DOMAINS": "allowed.example"}):
            # DNS解決は通る前提（外部アクセスしないようモック）
            def fake_getaddrinfo(host, *args, **kwargs):
                return [(2, None, None, None, ("93.184.216.34", 0))]

            with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
                with self.assertRaises(UrlValidationError):
                    validate_outbound_url("https://example.com/news", purpose="article")

    def test_rejects_ipv4_mapped_ipv6_loopback(self):
        # ::ffff:127.0.0.1 を返す想定
        def fake_getaddrinfo(host, *args, **kwargs):
            # AF_INET6 の sockaddr は (addr, port, flowinfo, scopeid)
            return [(10, None, None, None, ("::ffff:127.0.0.1", 0, 0, 0))]

        with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with self.assertRaises(UrlValidationError):
                validate_outbound_url("https://example.com/news", purpose="article")

    def test_rejects_redirect_to_blocked_when_redirects_enabled(self):
        # リダイレクトを許可した場合でも、遷移先のURL検証で拒否されること
        class FakeResponse:
            def __init__(self, status_code, headers):
                self.status_code = status_code
                self.headers = headers

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=65536):
                yield b""

            def close(self):
                return None

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, url, headers=None, timeout=None, stream=None, allow_redirects=None):
                self.calls += 1
                # 初回は302、以降は呼ばれない想定
                return FakeResponse(302, {"Location": "http://127.0.0.1/"})

        def validate_side_effect(url, *, purpose):
            if url.startswith("https://example.com"):
                return url
            raise UrlValidationError("blocked")

        with patch.dict(os.environ, {"URL_ALLOW_REDIRECTS": "1", "URL_MAX_REDIRECTS": "2"}):
            with patch("src.utils.security.requests.Session", return_value=FakeSession()):
                with patch("src.utils.security.validate_outbound_url", side_effect=validate_side_effect):
                    with self.assertRaises(UrlValidationError):
                        fetch_url_bytes("https://example.com/news", purpose="article")

    def test_rejects_when_content_length_over_limit(self):
        class FakeResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Type": "text/html; charset=utf-8", "Content-Length": "999"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=65536):
                yield b"x" * 10

            def close(self):
                return None

        class FakeSession:
            def get(self, url, headers=None, timeout=None, stream=None, allow_redirects=None):
                return FakeResponse()

        def ok_validate(url, *, purpose):
            return url

        with patch.dict(os.environ, {"HTTP_MAX_BYTES": "10"}):
            with patch("src.utils.security.requests.Session", return_value=FakeSession()):
                with patch("src.utils.security.validate_outbound_url", side_effect=ok_validate):
                    with self.assertRaises(ResponseTooLargeError):
                        fetch_url_bytes("https://example.com/news", purpose="article")

    def test_rejects_when_stream_exceeds_limit(self):
        class FakeResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Type": "text/html; charset=utf-8"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=65536):
                yield b"x" * 8
                yield b"y" * 8  # 合計16bytes

            def close(self):
                return None

        class FakeSession:
            def get(self, url, headers=None, timeout=None, stream=None, allow_redirects=None):
                return FakeResponse()

        def ok_validate(url, *, purpose):
            return url

        with patch.dict(os.environ, {"HTTP_MAX_BYTES": "10"}):
            with patch("src.utils.security.requests.Session", return_value=FakeSession()):
                with patch("src.utils.security.validate_outbound_url", side_effect=ok_validate):
                    with self.assertRaises(ResponseTooLargeError):
                        fetch_url_bytes("https://example.com/news", purpose="article")


if __name__ == "__main__":
    unittest.main()


