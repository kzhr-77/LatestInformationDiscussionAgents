import os
import unittest
from unittest.mock import patch

from src.utils.security import UrlValidationError, sanitize_url_for_logging, validate_outbound_url


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


if __name__ == "__main__":
    unittest.main()


