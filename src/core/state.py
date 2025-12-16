from typing import TypedDict, List, Optional
from src.models.schemas import Argument, Critique, FinalReport, Rebuttal

class DiscussionState(TypedDict, total=False):
    """
    LangGraphで共有する状態。

    Streamlit/UIから渡す初期状態は部分的（topic/messagesのみ）なため、
    total=False として「キーは存在しない可能性がある」前提に合わせる。
    """
    topic: str
    request_id: str
    halt: bool
    halt_reason: str
    article_text: str
    optimistic_argument: Optional[Argument]
    pessimistic_argument: Optional[Argument]
    critique: Optional[Critique]
    optimistic_rebuttal: Optional[Rebuttal]
    pessimistic_rebuttal: Optional[Rebuttal]
    final_report: Optional[FinalReport]
    messages: List[str]  # For history tracking

