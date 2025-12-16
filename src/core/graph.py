from langgraph.graph import StateGraph, END
from src.core.state import DiscussionState
from src.agents.researcher import ResearcherAgent
from src.agents.analyst_optimistic import OptimisticAnalystAgent
from src.agents.analyst_pessimistic import PessimisticAnalystAgent
from src.agents.fact_checker import FactCheckerAgent
from src.agents.reporter import ReporterAgent
from src.utils.llm import get_llm
from src.utils.llm_profiles import get_profile
from src.models.schemas import Argument, Critique, FinalReport, Rebuttal
import logging

def create_graph(model_name: str = "gemma3:4b"):
    """
    討論システムのグラフを作成する
    
    Args:
        model_name: 使用するLLMモデル名（デフォルト: gemma3:4b）
    
    Returns:
        コンパイル済みのStateGraph
    """
    logger = logging.getLogger(__name__)

    llm = get_llm(model_name, **get_profile("analysis").to_kwargs())
    llm_fact_checker = get_llm(model_name, **get_profile("fact_check").to_kwargs())
    
    # Initialize agents
    researcher = ResearcherAgent(llm)
    optimist = OptimisticAnalystAgent(llm)
    pessimist = PessimisticAnalystAgent(llm)
    checker = FactCheckerAgent(llm_fact_checker)
    reporter = ReporterAgent(llm)

    # Define nodes
    def research_node(state: DiscussionState):
        """フェーズ0: 記事取得ノード"""
        rid = state.get("request_id", "-")
        try:
            if not state.get("topic"):
                raise ValueError("トピックが指定されていません")
            article = researcher.run(state["topic"])
            if not article:
                raise ValueError("記事の取得に失敗しました")
            return {"article_text": article}
        except Exception as e:
            logger.exception("[%s] リサーチエラー: %s", rid, e)
            return {"article_text": f"エラー: {str(e)}"}

    def optimist_node(state: DiscussionState):
        """フェーズ1: 楽観的分析ノード"""
        rid = state.get("request_id", "-")
        try:
            if not state.get("article_text"):
                raise ValueError("記事テキストがありません")
            # LLMからArgument型を直接取得
            arg = optimist.analyze(state["article_text"])
            return {"optimistic_argument": arg}
        except Exception as e:
            logger.exception("[%s] 楽観的分析エラー: %s", rid, e)
            return {"optimistic_argument": Argument(
                conclusion=f"エラー: {str(e)}",
                evidence=[]
            )}

    def pessimist_node(state: DiscussionState):
        """フェーズ1: 悲観的分析ノード"""
        rid = state.get("request_id", "-")
        try:
            if not state.get("article_text"):
                raise ValueError("記事テキストがありません")
            # LLMからArgument型を直接取得
            arg = pessimist.analyze(state["article_text"])
            return {"pessimistic_argument": arg}
        except Exception as e:
            logger.exception("[%s] 悲観的分析エラー: %s", rid, e)
            return {"pessimistic_argument": Argument(
                conclusion=f"エラー: {str(e)}",
                evidence=[]
            )}
    
    def checker_node(state: DiscussionState):
        """フェーズ2: ファクトチェックノード"""
        rid = state.get("request_id", "-")
        try:
            optimistic_arg = state.get("optimistic_argument")
            pessimistic_arg = state.get("pessimistic_argument")
            article_text = state.get("article_text", "")
            
            if not optimistic_arg or not pessimistic_arg:
                raise ValueError("分析結果が不足しています")
            
            if not article_text:
                raise ValueError("記事テキストがありません")
            
            # LLMからCritique型を直接取得
            critique = checker.validate(optimistic_arg, pessimistic_arg, article_text)
            return {"critique": critique}
            
        except Exception as e:
            logger.exception("[%s] ファクトチェックエラー: %s", rid, e)
            return {"critique": Critique(
                bias_points=[],
                factual_errors=[f"エラー: {str(e)}"]
            )}

    def optimist_rebuttal_node(state: DiscussionState):
        """フェーズ3: 楽観的アナリスト反論ノード"""
        rid = state.get("request_id", "-")
        try:
            critique = state.get("critique")
            optimistic_arg = state.get("optimistic_argument")
            pessimistic_arg = state.get("pessimistic_argument")
            if not critique or not optimistic_arg or not pessimistic_arg:
                raise ValueError("反論に必要なデータ（critique/arguments）が不足しています")
            rebuttal = optimist.debate(
                critique=critique,
                opponent_argument=pessimistic_arg,
                original_argument=optimistic_arg,
            )
            return {"optimistic_rebuttal": rebuttal}
        except Exception as e:
            logger.exception("[%s] 楽観的反論エラー: %s", rid, e)
            return {"optimistic_rebuttal": Rebuttal(counter_points=[f"エラー: {str(e)}"], strengthened_evidence=[])}

    def pessimist_rebuttal_node(state: DiscussionState):
        """フェーズ3: 悲観的アナリスト反論ノード"""
        rid = state.get("request_id", "-")
        try:
            critique = state.get("critique")
            optimistic_arg = state.get("optimistic_argument")
            pessimistic_arg = state.get("pessimistic_argument")
            if not critique or not optimistic_arg or not pessimistic_arg:
                raise ValueError("反論に必要なデータ（critique/arguments）が不足しています")
            rebuttal = pessimist.debate(
                critique=critique,
                opponent_argument=optimistic_arg,
                original_argument=pessimistic_arg,
            )
            return {"pessimistic_rebuttal": rebuttal}
        except Exception as e:
            logger.exception("[%s] 悲観的反論エラー: %s", rid, e)
            return {"pessimistic_rebuttal": Rebuttal(counter_points=[f"エラー: {str(e)}"], strengthened_evidence=[])}

    def reporter_node(state: DiscussionState):
        """フェーズ4: レポート生成ノード"""
        rid = state.get("request_id", "-")
        try:
            report = reporter.create_report(state.get("messages", []))
            # Mocking return
            # TODO: 実際の実装では、LLMからFinalReport型を直接取得する
            # 現在はモックなので、最小限の構造を返す
            optimistic_arg = state.get("optimistic_argument") or Argument(conclusion="", evidence=[])
            pessimistic_arg = state.get("pessimistic_argument") or Argument(conclusion="", evidence=[])
            
            if isinstance(report, str):
                return {"final_report": FinalReport(
                    article_info="",
                    optimistic_view=optimistic_arg,
                    pessimistic_view=pessimistic_arg,
                    critique_points=[],
                    final_conclusion=report
                )}
            else:
                return {"final_report": FinalReport(
                    article_info=report.get("article_info", "") if isinstance(report, dict) else "",
                    optimistic_view=optimistic_arg,
                    pessimistic_view=pessimistic_arg,
                    critique_points=report.get("critique_points", []) if isinstance(report, dict) else [],
                    final_conclusion=report.get("final_conclusion", str(report)) if isinstance(report, dict) else str(report)
                )}
        except Exception as e:
            logger.exception("[%s] レポート生成エラー: %s", rid, e)
            optimistic_arg = state.get("optimistic_argument") or Argument(conclusion="", evidence=[])
            pessimistic_arg = state.get("pessimistic_argument") or Argument(conclusion="", evidence=[])
            return {"final_report": FinalReport(
                article_info="",
                optimistic_view=optimistic_arg,
                pessimistic_view=pessimistic_arg,
                critique_points=[],
                final_conclusion=f"エラー: {str(e)}"
            )}

    workflow = StateGraph(DiscussionState)

    # Add nodes
    workflow.add_node("researcher", research_node)
    workflow.add_node("optimist", optimist_node)
    workflow.add_node("pessimist", pessimist_node)
    workflow.add_node("checker", checker_node)
    workflow.add_node("optimist_rebuttal", optimist_rebuttal_node)
    workflow.add_node("pessimist_rebuttal", pessimist_rebuttal_node)
    workflow.add_node("reporter", reporter_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "optimist")
    workflow.add_edge("researcher", "pessimist")
    # AND合流: 楽観・悲観の両方が完了してからcheckerを1回だけ実行する
    workflow.add_edge(["optimist", "pessimist"], "checker")
    workflow.add_edge("checker", "optimist_rebuttal")
    workflow.add_edge("checker", "pessimist_rebuttal")
    # AND合流: 両反論が揃ってからレポートへ
    workflow.add_edge(["optimist_rebuttal", "pessimist_rebuttal"], "reporter")
    workflow.add_edge("reporter", END)

    return workflow.compile()

