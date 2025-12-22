import unittest
from unittest.mock import patch

from src.agents.researcher import ResearcherAgent
from src.utils.testing_models import AlwaysFailChatModel


class TestRssItemLinkPolicy(unittest.TestCase):
    def _rss_xml(self, item_link: str) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>テスト記事</title>
      <link>{item_link}</link>
      <description>テスト概要</description>
    </item>
  </channel>
</rss>
"""

    def test_policy_a_skips_non_same_domain_when_no_allowlist(self):
        agent = ResearcherAgent(AlwaysFailChatModel())
        agent.rss_feed_urls = ["https://feed.example.com/rss"]

        with patch("src.agents.researcher.fetch_feed_xml", return_value=self._rss_xml("https://other.example.org/a")):
            # A案: allowlist無しなら feed外ドメインはスキップ → 取得できず ValueError
            with patch.dict("os.environ", {"RSS_ITEM_LINK_POLICY": "A", "URL_ALLOWLIST_DOMAINS": ""}):
                with self.assertRaises(ValueError):
                    agent.run("テスト")

    def test_policy_b_allows_link_and_fetches(self):
        agent = ResearcherAgent(AlwaysFailChatModel())
        agent.rss_feed_urls = ["https://feed.example.com/rss"]

        with patch("src.agents.researcher.fetch_feed_xml", return_value=self._rss_xml("https://other.example.org/a")):
            with patch.dict("os.environ", {"RSS_ITEM_LINK_POLICY": "B", "URL_ALLOWLIST_DOMAINS": ""}):
                with patch.object(agent, "_fetch_from_url", return_value="本文") as m:
                    out = agent.run("テスト")
                    self.assertIn("[source] https://other.example.org/a", out)
                    self.assertIn("本文", out)
                    self.assertTrue(m.called)


if __name__ == "__main__":
    unittest.main()


