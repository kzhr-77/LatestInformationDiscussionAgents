from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from src.models.schemas import Argument, Critique
import json
import re
import logging

class FactCheckerAgent:
    """
    ファクトチェッカーエージェント
    
    楽観的・悲観的アナリストの主張を検証し、事実の正確性やバイアスを指摘する。
    """
    
    def __init__(self, model: BaseChatModel):
        """
        ファクトチェッカーエージェントを初期化
        
        Args:
            model: LLMモデル（温度パラメータは低めに設定することを推奨）
        """
        self.model = model
        self._init_prompts()
    
    def _init_prompts(self):
        """プロンプトテンプレートを初期化"""
        self.validate_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは客観的なファクトチェッカーです。楽観的アナリストと悲観的アナリストの主張を検証してください。

検証ポイント:
1. **引用の正確性**: 各アナリストが引用した部分が、元の記事の文脈に合っているか
2. **誇張の検出**: 記事の内容を過度に強調したり、歪曲していないか
3. **バイアスの特定**: 各アナリストが特定の視点に偏っていないか
4. **事実の確認**: 数値やデータが正確に引用されているか

重要ルール:
- 同じ文言/同じ意味の指摘を繰り返さない（言い換えも含む）
- 各項目は互いに重複しないようにする
- factual_errors の各項目は200文字以内にする

出力は以下の形式で構造化してください：
- bias_points: 各アナリストの主張における偏りやバイアスを指摘（楽観的アナリストと悲観的アナリストを分けて記述、各2-3個）
- factual_errors: 事実の誤りや文脈からの逸脱を指摘（具体的にどのアナリストのどの証拠に問題があるかを明記、2-4個）"""),
            ("human", """元の記事:
{article_text}

楽観的アナリストの主張:
結論: {optimistic_conclusion}
証拠:
{optimistic_evidence}

悲観的アナリストの主張:
結論: {pessimistic_conclusion}
証拠:
{pessimistic_evidence}

検証結果を出力してください。""")
        ])
    
    def validate(
        self, 
        optimistic_argument: Argument, 
        pessimistic_argument: Argument, 
        article_text: str
    ) -> Critique:
        """
        楽観的・悲観的アナリストの主張を検証する（フェーズ2）
        
        Args:
            optimistic_argument: 楽観的アナリストの主張
            pessimistic_argument: 悲観的アナリストの主張
            article_text: 元の記事テキスト（検証の参照用）
        
        Returns:
            Critique: バイアス指摘と事実誤りのリスト
        
        Raises:
            ValueError: 必要な引数が不足している場合
        """
        if not optimistic_argument or not pessimistic_argument:
            raise ValueError("検証する主張が不足しています。")
        if not article_text or not article_text.strip():
            raise ValueError("記事テキストが空です。")
        
        try:
            # プロンプトチェーンを作成
            # 温度を低めに設定したモデルを使用（事実検証のため）
            chain = self.validate_prompt | self.model.with_structured_output(Critique)
            
            # 証拠を整形
            optimistic_evidence_str = "\n".join([f"- {ev}" for ev in optimistic_argument.evidence])
            pessimistic_evidence_str = "\n".join([f"- {ev}" for ev in pessimistic_argument.evidence])
            
            # LLMを呼び出して構造化出力を取得
            result = chain.invoke({
                # 長文は先頭+末尾を残して文脈を保持（単純先頭切り捨てより誤判定が減る）
                "article_text": self._truncate_article_text(article_text),
                "optimistic_conclusion": optimistic_argument.conclusion,
                "optimistic_evidence": optimistic_evidence_str if optimistic_evidence_str else "（証拠なし）",
                "pessimistic_conclusion": pessimistic_argument.conclusion,
                "pessimistic_evidence": pessimistic_evidence_str if pessimistic_evidence_str else "（証拠なし）"
            })
            
            return self._normalize_critique(result)
            
        except Exception as e:
            # 構造化出力が崩れた場合は、JSON出力を強制して復旧を試みる
            logging.getLogger(__name__).exception("ファクトチェックエラー（structured_output）: %s", e)
            return self._fallback_validate_as_json(
                optimistic_argument=optimistic_argument,
                pessimistic_argument=pessimistic_argument,
                article_text=article_text,
                original_error=e,
            )

    def _normalize_critique(self, critique: Critique) -> Critique:
        """
        CritiqueをUI表示向けに正規化する。
        - factual_errors の各項目を200文字以内に丸める
        - 重複項目を除去する（LLMが同じ文を複数回出すケースの対策）
        """
        try:
            bias_points = list(getattr(critique, "bias_points", []) or [])
            factual_errors = list(getattr(critique, "factual_errors", []) or [])

            bias_points = self._dedupe_points(bias_points)
            factual_errors = [self._truncate_text(x, 200) for x in factual_errors]
            factual_errors = self._dedupe_points(factual_errors)

            return Critique(bias_points=bias_points, factual_errors=factual_errors)
        except Exception:
            # 失敗時は元のまま返す
            return critique

    @staticmethod
    def _dedupe_points(points: list[str]) -> list[str]:
        """
        表示用の重複除去。
        - 前後空白/連続空白を正規化
        - 「楽観的アナリスト:」「悲観的アナリスト:」「両アナリスト:」などのラベル差は比較時に無視
        """
        out: list[str] = []
        seen: set[str] = set()

        for p in points or []:
            raw = "" if p is None else str(p)
            raw = raw.strip()
            if not raw:
                continue

            key = raw
            # ラベル（比較時のみ除外）
            key = re.sub(r"^(楽観的アナリスト|悲観的アナリスト|両アナリスト)\s*[:：]\s*", "", key)
            # 空白正規化
            key = re.sub(r"\s+", " ", key).strip()

            if key in seen:
                continue
            seen.add(key)
            out.append(raw)

        return out

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        s = "" if text is None else str(text)
        s = s.strip()
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rstrip() + "…"

    def _truncate_article_text(self, article_text: str, max_chars: int = 8000) -> str:
        """
        記事テキストが長い場合に、先頭+末尾を残して短縮する。
        """
        text = (article_text or "").strip()
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-(max_chars // 2) :]
        return head + "\n\n...(中略)...\n\n" + tail

    def _fallback_validate_as_json(
        self,
        optimistic_argument: Argument,
        pessimistic_argument: Argument,
        article_text: str,
        original_error: Exception,
    ) -> Critique:
        """
        structured_outputが失敗した場合のフォールバック。
        LLMにJSON文字列で出させて、Pydantic(Critique)へ復元する。
        """
        try:
            optimistic_evidence_str = "\n".join([f"- {ev}" for ev in optimistic_argument.evidence])
            pessimistic_evidence_str = "\n".join([f"- {ev}" for ev in pessimistic_argument.evidence])

            prompt = ChatPromptTemplate.from_messages([
                ("system", "あなたは客観的なファクトチェッカーです。必ずJSONのみを出力してください。"),
                ("human", """以下を検証し、次のJSONのみを返してください。\n\nJSONスキーマ:\n{{\n  \"bias_points\": [\"...\"] ,\n  \"factual_errors\": [\"...\"]\n}}\n\n元の記事:\n{article_text}\n\n楽観的アナリスト:\n結論: {optimistic_conclusion}\n証拠:\n{optimistic_evidence}\n\n悲観的アナリスト:\n結論: {pessimistic_conclusion}\n証拠:\n{pessimistic_evidence}\n""")
            ])
            raw = (prompt | self.model).invoke({
                "article_text": self._truncate_article_text(article_text),
                "optimistic_conclusion": optimistic_argument.conclusion,
                "optimistic_evidence": optimistic_evidence_str if optimistic_evidence_str else "（証拠なし）",
                "pessimistic_conclusion": pessimistic_argument.conclusion,
                "pessimistic_evidence": pessimistic_evidence_str if pessimistic_evidence_str else "（証拠なし）",
            })

            # rawはMessage型になることがあるのでcontentを取り出す
            content = getattr(raw, "content", raw)
            if not isinstance(content, str):
                content = str(content)

            # JSON部分を抽出（前後に余計な文が付く場合に備える）
            match = re.search(r"\{[\s\S]*\}", content)
            json_text = match.group(0) if match else content
            data = json.loads(json_text)

            if hasattr(Critique, "model_validate"):
                critique = Critique.model_validate(data)  # pydantic v2
            else:
                critique = Critique.parse_obj(data)  # pydantic v1
            return self._normalize_critique(critique)

        except Exception as e:
            logging.getLogger(__name__).exception("ファクトチェックフォールバックエラー: %s", e)
            return Critique(
                bias_points=[
                    "検証に失敗しました（出力の構造化に失敗）。",
                    f"structured_outputエラー: {str(original_error)}",
                    f"fallbackエラー: {str(e)}",
                ],
                factual_errors=[],
            )

