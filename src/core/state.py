from typing import TypedDict, List, Optional
from src.models.schemas import Argument, Critique, FinalReport

class DiscussionState(TypedDict):
    topic: str
    article_text: str
    optimistic_argument: Optional[Argument]
    pessimistic_argument: Optional[Argument]
    critique: Optional[Critique]
    final_report: Optional[FinalReport]
    messages: List[str] # For history tracking

