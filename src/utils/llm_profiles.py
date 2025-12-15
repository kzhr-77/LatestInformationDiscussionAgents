from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProfile:
    temperature: float | None = None
    num_predict: int | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    stop: list[str] | None = None

    def to_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "stop": self.stop,
        }


PROFILES: dict[str, LLMProfile] = {
    # 既定
    "default": LLMProfile(temperature=0.7),
    # 分析（自由度は残すが、無駄な長文化を抑えたい場合はnum_predictを後で調整）
    "analysis": LLMProfile(temperature=0.7),
    # 事実検証（低温＋反復抑制）
    "fact_check": LLMProfile(
        temperature=0.3,
        repeat_penalty=1.15,
        repeat_last_n=128,
    ),
}


def get_profile(name: str) -> LLMProfile:
    return PROFILES.get(name, PROFILES["default"])


