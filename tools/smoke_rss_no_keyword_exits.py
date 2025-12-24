import sys
from pathlib import Path


class DummyResearcherNoKeyword:
    def run(self, topic: str) -> str:
        # ResearcherAgentと同じ例外型を投げる想定
        from src.agents.researcher import RssKeywordNotFoundError

        raise RssKeywordNotFoundError(f"RSSフィード内にキーワード '{topic}' の一致が見つかりませんでした。")


def main() -> int:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.core.orchestrator import OrchestrationAgent
    from src.utils.testing_models import AlwaysFailChatModel

    failing = AlwaysFailChatModel()
    orchestrator = OrchestrationAgent(llm=failing, llm_fact_checker=failing, researcher_agent=DummyResearcherNoKeyword())
    result = orchestrator.invoke({"topic": "テスト", "messages": [], "request_id": "smoke-no-keyword"})

    assert result.get("halt") is True
    assert "RSSフィード内にキーワード" in (result.get("halt_reason") or "")

    # 早期終了しているので後続成果物は無い（≒処理を終了）
    for k in ["optimistic_argument", "pessimistic_argument", "critique", "final_report"]:
        assert k not in result, f"should not have run downstream: {k}"

    print("OK: rss no-keyword -> halt and early END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


