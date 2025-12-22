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
- 出力には必ず **bias_points と factual_errors の両方**を含める（該当なしでも空配列 [] を入れる）

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
        
        # 案A: structured_output を使わず、常に JSON 文字列出力 → パースで復元する
        return self._fallback_validate_as_json(
            optimistic_argument=optimistic_argument,
            pessimistic_argument=pessimistic_argument,
            article_text=article_text,
            original_error=RuntimeError("CritiqueはJSON経由で復元（structured_output不使用）"),
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

            # --- 日本語化: まれに英語で返るケースがあるため、UI表示向けに日本語へ寄せる ---
            # - モデル未接続/失敗時はそのまま（フォールバック）
            bias_points = self._ensure_japanese_points(bias_points, kind="bias_points")
            factual_errors = self._ensure_japanese_points(factual_errors, kind="factual_errors")

            return Critique(bias_points=bias_points, factual_errors=factual_errors)
        except Exception:
            # 失敗時は元のまま返す
            return critique

    @staticmethod
    def _contains_japanese(text: str) -> bool:
        s = "" if text is None else str(text)
        # ひらがな・カタカナ・漢字が含まれていれば日本語っぽいとみなす
        return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", s))

    def _ensure_japanese_points(self, points: list[str], kind: str) -> list[str]:
        """
        bias_points / factual_errors に英語中心の項目が混ざる場合があるため、日本語へ寄せる。
        - 既に日本語っぽいものはそのまま
        - 翻訳に失敗した場合はそのまま（安全側）
        """
        items = [("" if x is None else str(x)).strip() for x in (points or [])]
        items = [x for x in items if x]
        if not items:
            return items

        # 英語中心と判断したものが無ければ何もしない
        needs = [x for x in items if not self._contains_japanese(x)]
        if not needs:
            return items

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "あなたは翻訳者です。必ず日本語で書き直してください。必ずJSONのみを出力してください。",
                    ),
                    (
                        "human",
                        """次の items を順番を変えずに日本語へ書き直してください。

ルール:
- 既に日本語の文はそのままでもよい
- 先頭に「楽観的アナリスト:」「悲観的アナリスト:」「両アナリスト:」「Optimistic Analyst:」等のラベルがある場合は、ラベル（コロンまで）を維持し、後続だけ日本語にする
- 各要素は200文字以内（超える場合は短く要約）
- 出力は必ずこのJSONスキーマ:
{{"items": ["..."]}}

items:
{items_json}
""",
                    ),
                ]
            )
            raw = (prompt | self.model).invoke({"items_json": json.dumps(items, ensure_ascii=False)})
            content = getattr(raw, "content", raw)
            if not isinstance(content, str):
                content = str(content)
            cleaned = self._strip_code_fences(content)
            json_text = (
                self._extract_first_json_object_stream(cleaned)
                or self._extract_first_json_object(cleaned)
                or cleaned
            )
            data = json.loads(json_text)
            out = data.get("items") if isinstance(data, dict) else None
            if not isinstance(out, list):
                return items
            out2 = [("" if x is None else str(x)).strip() for x in out]
            out2 = [x for x in out2 if x]
            # 長さが合わない場合は安全側（元を返す）
            if len(out2) != len(items):
                return items
            # 再度丸め・重複除去
            out2 = [self._truncate_text(x, 200) for x in out2]
            out2 = self._dedupe_points(out2)
            return out2
        except Exception as e:
            logging.getLogger(__name__).info("日本語化をスキップ（%s）: %s", kind, e)
            return items

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

            # --- JSON抽出の頑健化（案F1） ---
            # - ```json ... ``` のフェンス除去
            # - 複数JSONがある/前後に説明がある場合でも「最初にパースできたJSON」を採用
            cleaned = self._strip_code_fences(content)
            json_text = (
                self._extract_first_json_object_stream(cleaned)
                or self._extract_first_json_object(cleaned)
                or cleaned
            )
            data = json.loads(json_text)

            # 欠落/型崩れに備えて最低限の形へ整形
            if not isinstance(data, dict):
                data = {}
            bias_points = data.get("bias_points", [])
            factual_errors = data.get("factual_errors", [])
            if not isinstance(bias_points, list):
                bias_points = []
            if not isinstance(factual_errors, list):
                factual_errors = []
            data = {"bias_points": bias_points, "factual_errors": factual_errors}

            if hasattr(Critique, "model_validate"):
                critique = Critique.model_validate(data)  # pydantic v2
            else:
                critique = Critique.parse_obj(data)  # pydantic v1
            return self._normalize_critique(critique)

        except Exception as e:
            logging.getLogger(__name__).exception("ファクトチェックフォールバックエラー: %s", e)
            # 観測性: モデル出力の断片（記事本文ではなく、LLM出力側のみ）を短く残す
            try:
                snippet = self._safe_snippet(locals().get("content", ""), 480)
                if snippet:
                    logging.getLogger(__name__).warning("ファクトチェック復元失敗: model_output_snippet=%s", snippet)
            except Exception:
                pass
            return Critique(
                bias_points=[
                    "検証に失敗しました（出力の構造化に失敗）。",
                    f"structured_outputエラー: {str(original_error)}",
                    f"fallbackエラー: {str(e)}",
                ],
                factual_errors=[],
            )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        LLMが ```json ... ``` のようなフェンス付きで返した場合に除去する。
        """
        s = "" if text is None else str(text)
        s = s.strip()
        # 先頭・末尾のフェンスを軽く除去（中身に ``` が出るケースは稀なので単純化）
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s.strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        """
        文字列内から「最初に現れるJSONオブジェクト（{...}）」を抜き出す。
        - 非貪欲な正規表現で候補を拾い、最初にjson.loadsできたものを返す
        """
        s = "" if text is None else str(text)
        # 非貪欲に候補を列挙
        candidates = re.findall(r"\{[\s\S]*?\}", s)
        for c in candidates:
            try:
                json.loads(c)
                return c
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_first_json_object_stream(text: str) -> str | None:
        """
        文字列から最初のJSONオブジェクト（{...}）を括弧カウントで抽出する。
        - 正規表現より堅牢（ネストした{}や文字列内の{}を考慮）
        - 返すのは「最初に現れる開始{」から対応する閉じ}まで
        """
        s = "" if text is None else str(text)
        start = s.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            # 文字列の開始
            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # 開始がたまたまJSONでない場合は次の{を探す
                        nxt = s.find("{", start + 1)
                        if nxt < 0:
                            return None
                        start = nxt
                        depth = 0
                        in_str = False
                        esc = False
                        # i を start-1 に戻すのが理想だが、簡易にループを続けるため再帰で処理
                        return FactCheckerAgent._extract_first_json_object_stream(s[start:])
        return None

    @staticmethod
    def _safe_snippet(text: str, max_chars: int = 480) -> str:
        s = "" if text is None else str(text)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return ""
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rstrip() + "…"
