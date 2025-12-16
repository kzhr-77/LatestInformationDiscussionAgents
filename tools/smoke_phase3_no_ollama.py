import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.core.graph import create_graph
    from src.utils.testing_models import AlwaysFailChatModel

    failing = AlwaysFailChatModel()

    graph = create_graph(llm=failing, llm_fact_checker=failing)
    result = graph.invoke({"topic": "テストトピック", "messages": [], "request_id": "smoke"})

    required_keys = [
        "optimistic_argument",
        "pessimistic_argument",
        "critique",
        "optimistic_rebuttal",
        "pessimistic_rebuttal",
        "final_report",
    ]
    missing = [k for k in required_keys if k not in result]
    if missing:
        raise AssertionError(f"missing keys: {missing}")

    print("OK: phase3 smoke (no ollama) - keys present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


