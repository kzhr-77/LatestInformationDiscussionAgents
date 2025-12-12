from typing import List, Optional
from pydantic import BaseModel, Field

class Argument(BaseModel):
    conclusion: str
    evidence: List[str]

class Critique(BaseModel):
    bias_points: List[str]
    factual_errors: List[str]

class Rebuttal(BaseModel):
    counter_points: List[str] = Field(description="相手の主張への反論ポイント")
    strengthened_evidence: List[str] = Field(description="自分の主張を補強する追加証拠")

class FinalReport(BaseModel):
    article_info: str
    optimistic_view: Argument
    pessimistic_view: Argument
    critique_points: List[str]
    final_conclusion: str

