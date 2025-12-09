from typing import List, Optional
from pydantic import BaseModel, Field

class Argument(BaseModel):
    conclusion: str
    evidence: List[str]

class Critique(BaseModel):
    bias_points: List[str]
    factual_errors: List[str]

class FinalReport(BaseModel):
    article_info: str
    optimistic_view: Argument
    pessimistic_view: Argument
    critique_points: List[str]
    final_conclusion: str

