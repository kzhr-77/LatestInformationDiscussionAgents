from typing import List, Optional
from pydantic import BaseModel, Field

class Argument(BaseModel):
    conclusion: str
    evidence: List[str] = Field(default_factory=list)

class Critique(BaseModel):
    bias_points: List[str] = Field(default_factory=list)
    factual_errors: List[str] = Field(default_factory=list)

class Rebuttal(BaseModel):
    counter_points: List[str] = Field(default_factory=list, description="相手の主張への反論ポイント")
    strengthened_evidence: List[str] = Field(default_factory=list, description="自分の主張を補強する追加証拠")

class FinalReport(BaseModel):
    article_info: str
    optimistic_view: Argument
    pessimistic_view: Argument
    critique_points: List[str] = Field(default_factory=list)
    final_conclusion: str

