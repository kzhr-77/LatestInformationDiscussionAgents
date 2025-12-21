import unittest

from src.agents.reporter import ReporterAgent
from src.models.schemas import Argument, Critique, Rebuttal
from src.utils.testing_models import AlwaysFailChatModel


class TestReporterFallback(unittest.TestCase):
    def test_create_report_never_empty_and_contains_required_phrases(self):
        model = AlwaysFailChatModel()
        agent = ReporterAgent(model)

        article_text = "\n".join(
            [
                "[source] https://example.com/news",
                "[title] テスト記事タイトル",
                "",
                "政府は2025年12月に新制度を発表した。対象は全国の事業者。",
                "施行日は2026年4月1日で、移行期間は6カ月とされる。",
                "関係者は『準備が必要だ』と述べた。",
            ]
        )

        report = agent.create_report(
            article_text=article_text,
            optimistic_argument=Argument(conclusion="機会がある", evidence=["2026年4月1日に施行"]),
            pessimistic_argument=Argument(conclusion="リスクがある", evidence=["移行期間が6カ月"]),
            critique=Critique(
                bias_points=["楽観的アナリスト: 良い面に寄りすぎ"],
                factual_errors=["悲観的アナリスト: 引用が本文に無い可能性"],
            ),
            optimistic_rebuttal=Rebuttal(counter_points=["悲観は過度"], strengthened_evidence=[]),
            pessimistic_rebuttal=Rebuttal(counter_points=["楽観は根拠薄い"], strengthened_evidence=[]),
            article_url=None,
        )

        self.assertIn("タイトル:", report.article_info)
        self.assertIn("ソース:", report.article_info)
        self.assertIn("要約:", report.article_info)
        self.assertIn("確実度が高い点", report.final_conclusion)
        self.assertIn("不確かな点", report.final_conclusion)

        # critique_points はタグ付きで生成される
        self.assertTrue(any(p.startswith("[Factual]") for p in report.critique_points))
        self.assertTrue(any(p.startswith("[Bias]") for p in report.critique_points))


if __name__ == "__main__":
    unittest.main()


