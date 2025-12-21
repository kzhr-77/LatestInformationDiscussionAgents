import unittest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agents.fact_checker import FactCheckerAgent
from src.models.schemas import Argument


class FixedResponseChatModel(BaseChatModel):
    """決め打ちの文字列を返すテスト用ChatModel（structured_outputには依存しない想定）"""

    def __init__(self, content: str):
        super().__init__()
        self._content = content

    @property
    def _llm_type(self) -> str:
        return "fixed_response_chat_model"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = AIMessage(content=self._content)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class TestFactCheckerJsonParsing(unittest.TestCase):
    def test_parses_json_inside_code_fence_and_ignores_preamble(self):
        content = (
            "以下が検証結果です。\n"
            "```json\n"
            "{\n"
            '  "bias_points": ["楽観的アナリスト: 同じ主張の繰り返し", "楽観的アナリスト: 同じ主張の繰り返し"],\n'
            '  "factual_errors": ["悲観的アナリスト: ' + ("あ" * 260) + '"]\n'
            "}\n"
            "```\n"
            "（以上）"
        )
        model = FixedResponseChatModel(content)
        agent = FactCheckerAgent(model)

        critique = agent.validate(
            optimistic_argument=Argument(conclusion="結論A", evidence=["証拠1"]),
            pessimistic_argument=Argument(conclusion="結論B", evidence=["証拠2"]),
            article_text="元記事テキスト",
        )

        # bias_points は重複が除去される
        self.assertEqual(len(critique.bias_points), 1)
        # factual_errors は200文字に丸められる（末尾が…）
        self.assertTrue(len(critique.factual_errors[0]) <= 201)
        self.assertTrue(critique.factual_errors[0].endswith("…"))

    def test_picks_first_parseable_json_object_when_multiple_exist(self):
        content = (
            "説明文\n"
            "{ invalid json }\n"
            "{"
            '  "bias_points": ["x"],'
            '  "factual_errors": ["y"]'
            "}\n"
            "{"
            '  "bias_points": ["later"],'
            '  "factual_errors": ["later"]'
            "}"
        )
        model = FixedResponseChatModel(content)
        agent = FactCheckerAgent(model)

        critique = agent.validate(
            optimistic_argument=Argument(conclusion="A", evidence=[]),
            pessimistic_argument=Argument(conclusion="B", evidence=[]),
            article_text="元記事テキスト",
        )

        self.assertEqual(critique.bias_points, ["x"])
        self.assertEqual(critique.factual_errors, ["y"])


if __name__ == "__main__":
    unittest.main()


