from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class AlwaysFailChatModel(BaseChatModel):
    """
    外部LLM/Ollamaに依存しないスモークテスト用のモデル。
    invoke/generate系を必ず失敗させ、各Agentの例外フォールバック経路で完走できるかを確認する。
    """

    def __init__(self, error_message: str = "AlwaysFailChatModel: forced failure", **kwargs: Any):
        super().__init__(**kwargs)
        self._error_message = error_message

    @property
    def _llm_type(self) -> str:  # BaseChatModelの要件
        return "always_fail_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(self._error_message)

    # 念のため（内部で呼ばれる可能性があるため）
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(self._error_message)


