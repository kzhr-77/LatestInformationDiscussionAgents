from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.agents.analyst_optimistic import OptimisticAnalystAgent
from src.agents.analyst_pessimistic import PessimisticAnalystAgent
from src.agents.fact_checker import FactCheckerAgent
from src.agents.reporter import ReporterAgent
from src.agents.researcher import ResearcherAgent, RssKeywordNotFoundError
from src.core.state import DiscussionState
from src.models.schemas import Argument, Critique, FinalReport, Rebuttal
from src.utils.llm import get_llm
from src.utils.llm_profiles import get_profile


@dataclass(frozen=True)
class OrchestrationOptions:
    """
    オーケストレーションの挙動スイッチ。

    Note:
    - A案（推奨）: OrchestrationAgentは「結論の文章生成」は行わず、
      各エージェント出力を“素材として確定”し ReporterAgent に渡す。
    """

    truncate_for_prompt_chars: int = 4000
    truncate_article_for_report_chars: int = 8000


class OrchestrationAgent:
    """
    LangGraph(StateGraph) の代替となるオーケストレーション専用エージェント。

    役割:
    - ResearcherAgent に議題（記事本文）取得を依頼
    - 楽観/悲観アナリストの進行（主張生成・反論生成）
    - FactChecker を呼び、批評(Critique)を取得
    - （A案）結論は“素材として確定”し、ReporterAgent にレポート作成を依頼
    """

    def __init__(
        self,
        model_name: str = "gemma3:4b",
        *,
        llm=None,
        llm_fact_checker=None,
        researcher_agent=None,
        options: OrchestrationOptions | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.options = options or OrchestrationOptions()

        # 通常はOllamaでLLMを生成するが、テスト/スモーク用途では外部から注入できるようにする
        if llm is None:
            llm = get_llm(model_name, verify_model=False, **get_profile("analysis").to_kwargs())
        if llm_fact_checker is None:
            llm_fact_checker = get_llm(model_name, verify_model=False, **get_profile("fact_check").to_kwargs())

        # Initialize agents
        self.researcher = researcher_agent or ResearcherAgent(llm)
        self.optimist = OptimisticAnalystAgent(llm)
        self.pessimist = PessimisticAnalystAgent(llm)
        self.checker = FactCheckerAgent(llm_fact_checker)
        self.reporter = ReporterAgent(llm)

    @staticmethod
    def _truncate_for_prompt(text: str, max_chars: int) -> str:
        s = (text or "").strip()
        if len(s) <= max_chars:
            return s
        head = s[: max_chars // 2]
        tail = s[-(max_chars // 2) :]
        return head + "\n\n...(中略)...\n\n" + tail

    def invoke(self, initial_state: DiscussionState) -> DiscussionState:
        """
        LangGraph の graph.invoke(...) 互換の実行メソッド。

        返り値は DiscussionState を拡張した dict（既存UI/スモーク互換）とする。
        """
        state: DiscussionState = dict(initial_state or {})
        rid = state.get("request_id", "-")

        # ---- Phase0: Research ----
        try:
            if not state.get("topic"):
                raise ValueError("トピックが指定されていません")
            article = self.researcher.run(state["topic"])
            if not article:
                raise ValueError("記事の取得に失敗しました")
            state["article_text"] = article
        except RssKeywordNotFoundError as e:
            self.logger.info("[%s] RSSキーワード一致なし: %s", rid, e)
            state["halt"] = True
            state["halt_reason"] = str(e)
            return state
        except Exception as e:
            self.logger.exception("[%s] リサーチエラー: %s", rid, e)
            state["article_text"] = f"エラー: {str(e)}"

        # ---- Guard: early exit ----
        if state.get("halt"):
            return state

        article_text = state.get("article_text") or ""

        # ---- Phase1: Analysts ----
        try:
            if state.get("optimistic_argument") is None:
                if not article_text:
                    raise ValueError("記事テキストがありません")
                state["optimistic_argument"] = self.optimist.analyze(article_text)
        except Exception as e:
            self.logger.exception("[%s] 楽観的分析エラー: %s", rid, e)
            state["optimistic_argument"] = Argument(conclusion=f"エラー: {str(e)}", evidence=[])

        try:
            if state.get("pessimistic_argument") is None:
                if not article_text:
                    raise ValueError("記事テキストがありません")
                state["pessimistic_argument"] = self.pessimist.analyze(article_text)
        except Exception as e:
            self.logger.exception("[%s] 悲観的分析エラー: %s", rid, e)
            state["pessimistic_argument"] = Argument(conclusion=f"エラー: {str(e)}", evidence=[])

        optimistic_arg = state.get("optimistic_argument") or Argument(conclusion="", evidence=[])
        pessimistic_arg = state.get("pessimistic_argument") or Argument(conclusion="", evidence=[])

        # ---- Phase2: Fact check ----
        try:
            if state.get("critique") is None:
                if not article_text:
                    raise ValueError("記事テキストがありません")
                critique = self.checker.validate(optimistic_arg, pessimistic_arg, article_text)
                state["critique"] = critique
        except Exception as e:
            self.logger.exception("[%s] ファクトチェックエラー: %s", rid, e)
            state["critique"] = Critique(bias_points=[], factual_errors=[f"エラー: {str(e)}"])

        critique = state.get("critique") or Critique(bias_points=[], factual_errors=[])

        # ---- Phase3: Rebuttals ----
        try:
            if state.get("optimistic_rebuttal") is None:
                rebuttal = self.optimist.debate(
                    critique=critique,
                    opponent_argument=pessimistic_arg,
                    original_argument=optimistic_arg,
                    article_text=self._truncate_for_prompt(article_text, self.options.truncate_for_prompt_chars),
                )
                state["optimistic_rebuttal"] = rebuttal
        except Exception as e:
            self.logger.exception("[%s] 楽観的反論エラー: %s", rid, e)
            state["optimistic_rebuttal"] = Rebuttal(counter_points=[f"エラー: {str(e)}"], strengthened_evidence=[])

        try:
            if state.get("pessimistic_rebuttal") is None:
                rebuttal = self.pessimist.debate(
                    critique=critique,
                    opponent_argument=optimistic_arg,
                    original_argument=pessimistic_arg,
                    article_text=self._truncate_for_prompt(article_text, self.options.truncate_for_prompt_chars),
                )
                state["pessimistic_rebuttal"] = rebuttal
        except Exception as e:
            self.logger.exception("[%s] 悲観的反論エラー: %s", rid, e)
            state["pessimistic_rebuttal"] = Rebuttal(counter_points=[f"エラー: {str(e)}"], strengthened_evidence=[])

        optimistic_rebuttal = state.get("optimistic_rebuttal") or Rebuttal(counter_points=[], strengthened_evidence=[])
        pessimistic_rebuttal = state.get("pessimistic_rebuttal") or Rebuttal(counter_points=[], strengthened_evidence=[])

        # ---- Phase4: Report ----
        try:
            if state.get("final_report") is None:
                final_report = self.reporter.create_report(
                    article_text=self._truncate_for_prompt(article_text, self.options.truncate_article_for_report_chars),
                    optimistic_argument=optimistic_arg,
                    pessimistic_argument=pessimistic_arg,
                    critique=critique,
                    optimistic_rebuttal=optimistic_rebuttal,
                    pessimistic_rebuttal=pessimistic_rebuttal,
                    article_url=state.get("topic"),
                )
                state["final_report"] = final_report
        except Exception as e:
            self.logger.exception("[%s] レポート生成エラー: %s", rid, e)
            state["final_report"] = FinalReport(
                article_info="",
                optimistic_view=optimistic_arg,
                pessimistic_view=pessimistic_arg,
                critique_points=[],
                final_conclusion=f"エラー: {str(e)}",
            )

        return state


