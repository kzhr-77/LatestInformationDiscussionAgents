from __future__ import annotations

import logging
import re
import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models.schemas import Argument, Critique, FinalReport, Rebuttal


class ReporterAgent:
    """
    レポートエージェント（フェーズ4）

    全フェーズの出力（主張、批判、反論）を統合し、FinalReportを生成する。
    """

    def __init__(self, model: BaseChatModel):
        self.model = model
        self._init_prompts()

    def _init_prompts(self) -> None:
        # 1) 事実抽出（本文から「確実に言える点」だけ抽出）
        self.facts_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """あなたはレポートエージェントです。記事本文から「確実に言える事実」を抽出してください。

重要ルール:
- 出力は**必ず日本語**
- 記事本文に無い事実を作らない（推測は禁止）
- できるだけ数字/固有名詞/決定事項を含める
- 可能なら「本文からの引用候補（抜粋）」に含まれる表現を短く含める（根拠の手がかり）

出力は次の構造（ExtractedFacts）に合わせること:
- key_facts: 箇条書き（5〜10個、各200文字以内、重複禁止）
- unknowns: 不明点/本文から断定できない点（2〜6個）""",
                ),
                (
                    "human",
                    """記事タイトル:
{article_title}

ソースURL:
{article_url}

記事本文（抜粋）:
{article_text}

本文からの引用候補（抜粋）:
{article_quotes}

上記に基づき、事実抽出をしてください。""",
                ),
            ]
        )
        # facts: JSON文字列フォールバック（structured_outputが使えない/壊れるモデル向け）
        self.facts_prompt_json = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはレポートエージェントです。必ずJSONのみを出力してください。"),
                (
                    "human",
                    """次のJSONのみを返してください。\n\nJSONスキーマ:\n{{\n  \"key_facts\": [\"...\"] ,\n  \"unknowns\": [\"...\"]\n}}\n\n記事タイトル:\n{article_title}\n\nソースURL:\n{article_url}\n\n記事本文（抜粋）:\n{article_text}\n\n本文からの引用候補（抜粋）:\n{article_quotes}\n""",
                ),
            ]
        )

        # 2) 統合（抽出した事実 + 各エージェント出力を統合）
        self.report_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """あなたはレポートエージェントです。抽出済みの事実と討論の出力を統合し、最終要約と統合結論を作成してください。

重要ルール:
- 出力は**必ず日本語**
- 記事本文に無い事実を作らない（不明な点は「不明」と書く）
- 一般論だけで終わらせない。下の「抽出済み事実」または「本文引用候補」に含まれる具体情報に必ず触れること
- 要約/結論の中で、少なくとも2点は「抽出済み事実」または「本文引用候補」の文言を短く引用して含めること（10〜25文字程度の断片でよい）
- 要約/結論に**新しい数字/固有名詞/断定的な因果**を追加しない（本文に無い情報は「不明」「可能性」にとどめる）
- 「引用根拠チェック」で一致しない可能性がある点は、事実として断定せず注意喚起として扱う
- 結論は「機会」と「リスク」を両方扱う

出力は次の構造（ReportContent）に合わせること:
- summary: 記事内容の要約（2〜5文）。少なくとも2つは具体情報（数字/固有名詞/決定事項）に触れる。
- final_conclusion: 議論を踏まえた統合結論（2〜6文）。最後に必ず「確実度が高い点: ...」「不確かな点: ...」を1文ずつ含める。""",
                ),
                (
                    "human",
                    """記事タイトル:
{article_title}

ソースURL:
{article_url}

抽出済み事実:
{extracted_facts}

不明点（本文から断定できない点）:
{unknowns}

本文引用候補（抜粋）:
{article_quotes}

楽観的アナリストの主張:
{optimistic_argument}

悲観的アナリストの主張:
{pessimistic_argument}

ファクトチェッカーの批評:
{critique}

楽観的アナリストの反論:
{optimistic_rebuttal}

悲観的アナリストの反論:
{pessimistic_rebuttal}

引用根拠チェック（本文に見当たらない可能性）:
{evidence_mismatch_notes}

要約（summary）と統合結論（final_conclusion）を生成してください。""",
                ),
            ]
        )
        # report: JSON文字列フォールバック
        self.report_prompt_json = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはレポートエージェントです。必ずJSONのみを出力してください。"),
                (
                    "human",
                    """次のJSONのみを返してください。\n\nJSONスキーマ:\n{{\n  \"summary\": \"...\" ,\n  \"final_conclusion\": \"...\"\n}}\n\n記事タイトル:\n{article_title}\n\nソースURL:\n{article_url}\n\n抽出済み事実:\n{extracted_facts}\n\n不明点:\n{unknowns}\n\n本文引用候補（抜粋）:\n{article_quotes}\n\n楽観的アナリストの主張:\n{optimistic_argument}\n\n悲観的アナリストの主張:\n{pessimistic_argument}\n\nファクトチェッカーの批評:\n{critique}\n\n楽観的アナリストの反論:\n{optimistic_rebuttal}\n\n悲観的アナリストの反論:\n{pessimistic_rebuttal}\n\n引用根拠チェック:\n{evidence_mismatch_notes}\n""",
                ),
            ]
        )

    @staticmethod
    def _truncate(text: str, max_chars: int = 8000) -> str:
        s = (text or "").strip()
        if len(s) <= max_chars:
            return s
        head = s[: max_chars // 2]
        tail = s[-(max_chars // 2) :]
        return head + "\n\n...(中略)...\n\n" + tail

    @staticmethod
    def _extract_article_header(article_text: str, fallback_url: str | None = None) -> tuple[str, str, str]:
        """
        ResearcherAgent(_search_with_rss)のヘッダ形式:
        [source] URL
        [title] タイトル

        を優先して抽出する。無い場合はタイトル不明、URLはfallback_urlを使う。
        """
        text = (article_text or "").strip()
        url = ""
        title = ""
        body = text

        m_url = re.search(r"^\\[source\\]\\s*(.+)$", text, flags=re.MULTILINE)
        if m_url:
            url = m_url.group(1).strip()
        m_title = re.search(r"^\\[title\\]\\s*(.+)$", text, flags=re.MULTILINE)
        if m_title:
            title = m_title.group(1).strip()

        # ヘッダっぽい行を先頭から取り除く
        lines = text.splitlines()
        filtered = []
        for ln in lines:
            if ln.startswith("[source]") or ln.startswith("[title]"):
                continue
            filtered.append(ln)
        body = "\n".join(filtered).strip()

        if not url and fallback_url:
            url = fallback_url
        if not title:
            title = "（不明）"
        if not url:
            url = "（不明）"
        return title, url, body

    @staticmethod
    def _fmt_argument(arg: Argument) -> str:
        conclusion = "" if arg is None else str(getattr(arg, "conclusion", "") or "")
        evidence = getattr(arg, "evidence", []) if arg is not None else []
        if evidence is None:
            evidence = []
        ev = "\n".join([f"- {x}" for x in evidence]) if evidence else "（証拠なし）"
        return f"結論: {conclusion}\n証拠:\n{ev}"

    @staticmethod
    def _fmt_rebuttal(rb: Rebuttal) -> str:
        cps = getattr(rb, "counter_points", []) if rb is not None else []
        ses = getattr(rb, "strengthened_evidence", []) if rb is not None else []
        if cps is None:
            cps = []
        if ses is None:
            ses = []
        cp = "\n".join([f"- {x}" for x in cps]) if cps else "（なし）"
        se = "\n".join([f"- {x}" for x in ses]) if ses else "（なし）"
        return f"反論ポイント:\n{cp}\n補強証拠:\n{se}"

    @staticmethod
    def _fmt_critique(c: Critique) -> str:
        bias = getattr(c, "bias_points", []) if c is not None else []
        factual = getattr(c, "factual_errors", []) if c is not None else []
        if bias is None:
            bias = []
        if factual is None:
            factual = []
        b = "\n".join([f"- {x}" for x in bias]) if bias else "（なし）"
        f = "\n".join([f"- {x}" for x in factual]) if factual else "（なし）"
        return f"バイアス指摘:\n{b}\n事実誤り:\n{f}"

    @staticmethod
    def _evidence_mismatch_notes(article_text: str, optimistic_argument: Argument, pessimistic_argument: Argument) -> str:
        """
        アナリストの証拠(evidence)が記事本文に“文字列として”存在するかを簡易チェックする。
        一致しない場合はレポートに注意点として渡す。
        """
        text = (article_text or "")
        out: list[str] = []

        def check(label: str, arg: Argument) -> None:
            evs = list(getattr(arg, "evidence", []) or [])
            misses = [ev for ev in evs if ev and ev not in text]
            if misses:
                # 長文化を避ける
                for ev in misses[:5]:
                    out.append(f"{label}: 本文に一致する引用が見当たらない可能性: {ev}")

        check("楽観", optimistic_argument)
        check("悲観", pessimistic_argument)
        return "\n".join([f"- {x}" for x in out]) if out else "（なし）"

    @staticmethod
    def _pick_article_quotes(article_body: str, limit: int = 6) -> str:
        """
        本文から「引用候補」を機械的に抜粋する。
        - 長すぎる/短すぎる行は除外
        - 数字/日付/単位がある行を優先
        """
        body = (article_body or "").strip()
        # まずは改行ベース（見出し/箇条書きがあるケースに強い）
        lines = [re.sub(r"\s+", " ", (ln or "")).strip() for ln in body.splitlines()]
        lines = [ln for ln in lines if 20 <= len(ln) <= 180]

        # 改行が少ない記事は1行が長くなりやすいので、文分割を追加（軽量な日本語句点ベース）
        if len(lines) < max(3, limit // 2) and len(body) > 200:
            # 「。！？？」でざっくり区切る（句点を残す）
            parts = re.split(r"(?<=[。！？\?])", re.sub(r"\s+", " ", body))
            sents = [p.strip() for p in parts if p and p.strip()]
            sents = [s for s in sents if 20 <= len(s) <= 180]
            lines.extend(sents)
            # 再度長さフィルタ（念のため）
            lines = [ln for ln in lines if 20 <= len(ln) <= 180]
        # 重複除去（先勝ち）
        uniq: list[str] = []
        seen: set[str] = set()
        for ln in lines:
            if ln in seen:
                continue
            seen.add(ln)
            uniq.append(ln)

        def score(s: str) -> int:
            sc = 0
            if re.search(r"\d", s):
                sc += 3
            if any(tok in s for tok in ["年", "月", "日", "円", "%", "％", "兆", "億"]):
                sc += 2
            if len(s) >= 60:
                sc += 1
            return sc

        ranked = sorted(uniq, key=score, reverse=True)
        picked = ranked[:limit] if ranked else uniq[:limit]
        if not picked:
            return "（本文から抽出できませんでした）"
        return "\n".join([f"- {x}" for x in picked])

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        s = "" if text is None else str(text)
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s.strip()

    @staticmethod
    def _extract_first_json_object_stream(text: str) -> str | None:
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
                        return None
        return None

    @staticmethod
    def _facts_looks_weak(extracted_facts: list[str], quote_lines: list[str]) -> bool:
        facts = [("" if x is None else str(x)).strip() for x in (extracted_facts or [])]
        facts = [x for x in facts if x]
        if len(facts) < 3 and quote_lines:
            return True

        # 具体性のシグナル（数字/単位）
        specific = 0
        for f in facts[:8]:
            if re.search(r"\d", f) or any(tok in f for tok in ["年", "月", "日", "円", "%", "％", "兆", "億", "万人", "社", "件"]):
                specific += 1
        if specific == 0 and quote_lines:
            return True

        # 一般論が多い（ざっくり）
        generic_tokens = ["一般的に", "重要", "必要", "求められる", "注目", "議論", "影響", "可能性", "慎重"]
        genericish = sum(1 for f in facts[:8] if any(t in f for t in generic_tokens))
        if genericish >= 4 and quote_lines:
            return True

        return False

    @staticmethod
    def _grounding_score(text: str, anchors: list[str]) -> int:
        """
        本文由来アンカー（抽出事実/引用候補）にどれだけ寄っているかの簡易スコア。
        - 文字列一致の弱いヒューリスティックだが、一般論化の検知に有効
        """
        s = (text or "").strip()
        if not s:
            return 0
        sc = 0
        if re.search(r"\d", s):
            sc += 2
        # アンカー断片（先頭15〜25文字）を含むか
        for a in (anchors or [])[:8]:
            a2 = ("" if a is None else str(a)).strip()
            if not a2:
                continue
            frag = a2[:20]
            if frag and frag in s:
                sc += 2
        # 一般論語の多さで減点
        generic_tokens = ["一般的に", "重要", "必要", "求められる", "注目", "議論", "影響", "可能性", "慎重", "べき"]
        genericish = sum(1 for t in generic_tokens if t in s)
        if genericish >= 4:
            sc -= 2
        return sc

    @staticmethod
    def _synthesize_summary_from_facts(extracted_facts: list[str], quote_lines: list[str]) -> str:
        """
        LLM出力が一般論化したときの、本文ベース最小要約。
        """
        facts = [("" if x is None else str(x)).strip() for x in (extracted_facts or [])]
        facts = [x for x in facts if x]
        if not facts and quote_lines:
            facts = quote_lines[:3]
        top = facts[:3]
        if not top:
            return "この記事は本文から具体情報を十分に抽出できませんでした（URL/サイトの取得制限や本文構造の影響の可能性）。"
        inline = " / ".join([x[:80] + ("…" if len(x) > 80 else "") for x in top])
        return f"この記事は本文から次の点が確認できます: {inline}"

    @staticmethod
    def _synthesize_conclusion_from_facts(
        extracted_facts: list[str],
        unknowns: list[str],
        critique_points: list[str],
        quote_lines: list[str],
        has_mismatch: bool,
    ) -> str:
        """
        LLM出力が弱い/壊れたときの、本文ベース最小結論（機会/リスク/確実&不確か を含む）。
        """
        facts = [("" if x is None else str(x)).strip() for x in (extracted_facts or []) if ("" if x is None else str(x)).strip()]
        unks = [("" if x is None else str(x)).strip() for x in (unknowns or []) if ("" if x is None else str(x)).strip()]
        q = quote_lines[0][:80] + ("…" if quote_lines and len(quote_lines[0]) > 80 else "") if quote_lines else ""
        hi = f"本文抜粋（「{q}」）に基づく範囲の事実。" if q else "本文から直接確認できる範囲の事実。"
        if has_mismatch:
            lo = "アナリストの引用の一部は本文一致しない可能性があり、追加検証が必要。"
        else:
            lo = (unks[0] if unks else "記事本文だけでは影響評価や因果の断定が難しい点。")
        # まずは本文から言える範囲で「機会/リスク」を分ける（汎用だが断定は避ける）
        opp_anchor = facts[0][:60] + ("…" if facts and len(facts[0]) > 60 else "") if facts else ""
        risk_anchor = facts[1][:60] + ("…" if len(facts) > 1 and len(facts[1]) > 60 else "") if len(facts) > 1 else ""
        caution = ""
        if critique_points:
            caution = f"（留意: {critique_points[0][:120]}）"
        return (
            f"抽出できた事実の範囲で見ると、機会は「{opp_anchor}」のような動きが実現した場合に期待される点として整理できます。"
            f"一方、リスクは「{risk_anchor}」など不確実性や副作用を含む可能性があるため、断定せず追加確認が必要です。{caution} "
            f"確実度が高い点: {hi} 不確かな点: {lo}"
        ).strip()

    def create_report(
        self,
        article_text: str,
        optimistic_argument: Argument,
        pessimistic_argument: Argument,
        critique: Critique,
        optimistic_rebuttal: Rebuttal,
        pessimistic_rebuttal: Rebuttal,
        article_url: Optional[str] = None,
    ) -> FinalReport:
        """
        フェーズ4: 最終レポートを生成する。
        - optimistic_view / pessimistic_view は state の値をそのまま採用（幻覚の混入を避ける）
        - LLMは summary / final_conclusion のみ生成
        """
        try:
            title, url, body = self._extract_article_header(article_text, fallback_url=article_url)

            quote_lines = [ln.strip()[2:].strip() for ln in self._pick_article_quotes(body, limit=6).splitlines() if ln.strip().startswith("- ")]

            # 1) 事実抽出（本文ベース）: 失敗しても機械抽出で続行（案R1）
            extracted_facts: list[str] = []
            unknowns: list[str] = []
            try:
                facts_chain = self.facts_prompt | self.model.with_structured_output(ExtractedFacts)
                extracted: ExtractedFacts = facts_chain.invoke(
                    {
                        "article_title": title,
                        "article_url": url,
                        "article_text": self._truncate(body, 8000),
                        "article_quotes": "\n".join([f"- {x}" for x in quote_lines]) if quote_lines else "（抽出できませんでした）",
                    }
                )
                extracted_facts = list(getattr(extracted, "key_facts", []) or [])
                unknowns = list(getattr(extracted, "unknowns", []) or [])
            except Exception as e:
                logging.getLogger(__name__).exception("事実抽出エラー（フォールバックへ切替）: %s", e)
                # 1-b) JSON文字列フォールバック（structured_output未対応/不安定なモデル向け）
                try:
                    raw = (self.facts_prompt_json | self.model).invoke(
                        {
                            "article_title": title,
                            "article_url": url,
                            "article_text": self._truncate(body, 8000),
                            "article_quotes": "\n".join([f"- {x}" for x in quote_lines]) if quote_lines else "（抽出できませんでした）",
                        }
                    )
                    content = getattr(raw, "content", raw)
                    if not isinstance(content, str):
                        content = str(content)
                    cleaned = self._strip_code_fences(content)
                    json_text = self._extract_first_json_object_stream(cleaned) or cleaned
                    data = json.loads(json_text)
                    if not isinstance(data, dict):
                        data = {}
                    extracted_facts = list(data.get("key_facts", []) or [])
                    unknowns = list(data.get("unknowns", []) or [])
                except Exception:
                    # 機械抽出: 引用候補をそのまま事実候補として利用
                    extracted_facts = quote_lines[:8] if quote_lines else []
                    unknowns = [
                        "記事本文だけでは影響評価や因果の断定が難しい点がある可能性。",
                        "アナリストの主張の一部は本文の直接引用ではない可能性。",
                    ]

            # facts品質が弱い場合は、引用候補へ寄せる（モデルの一般論化対策）
            if self._facts_looks_weak(extracted_facts, quote_lines):
                extracted_facts = quote_lines[:8] if quote_lines else extracted_facts

            extracted_facts_text = "\n".join([f"- {x}" for x in extracted_facts]) if extracted_facts else "（抽出できませんでした）"
            unknowns_text = "\n".join([f"- {x}" for x in unknowns]) if unknowns else "（なし）"

            # 2) 統合（討論の出力も考慮）
            content: ReportContent | None = None
            try:
                report_chain = self.report_prompt | self.model.with_structured_output(ReportContent)
                content = report_chain.invoke(
                    {
                        "article_title": title,
                        "article_url": url,
                        "extracted_facts": extracted_facts_text,
                        "unknowns": unknowns_text,
                        "article_quotes": "\n".join([f"- {x}" for x in quote_lines]) if quote_lines else "（抽出できませんでした）",
                        "optimistic_argument": self._fmt_argument(optimistic_argument),
                        "pessimistic_argument": self._fmt_argument(pessimistic_argument),
                        "critique": self._fmt_critique(critique),
                        "optimistic_rebuttal": self._fmt_rebuttal(optimistic_rebuttal),
                        "pessimistic_rebuttal": self._fmt_rebuttal(pessimistic_rebuttal),
                        "evidence_mismatch_notes": self._evidence_mismatch_notes(article_text, optimistic_argument, pessimistic_argument),
                    }
                )
            except Exception as e:
                logging.getLogger(__name__).exception("統合レポート生成エラー（テンプレで復旧）: %s", e)
                # 2-b) JSON文字列フォールバック
                try:
                    raw = (self.report_prompt_json | self.model).invoke(
                        {
                            "article_title": title,
                            "article_url": url,
                            "extracted_facts": extracted_facts_text,
                            "unknowns": unknowns_text,
                            "article_quotes": "\n".join([f"- {x}" for x in quote_lines]) if quote_lines else "（抽出できませんでした）",
                            "optimistic_argument": self._fmt_argument(optimistic_argument),
                            "pessimistic_argument": self._fmt_argument(pessimistic_argument),
                            "critique": self._fmt_critique(critique),
                            "optimistic_rebuttal": self._fmt_rebuttal(optimistic_rebuttal),
                            "pessimistic_rebuttal": self._fmt_rebuttal(pessimistic_rebuttal),
                            "evidence_mismatch_notes": self._evidence_mismatch_notes(article_text, optimistic_argument, pessimistic_argument),
                        }
                    )
                    content_s = getattr(raw, "content", raw)
                    if not isinstance(content_s, str):
                        content_s = str(content_s)
                    cleaned = self._strip_code_fences(content_s)
                    json_text = self._extract_first_json_object_stream(cleaned) or cleaned
                    data = json.loads(json_text)
                    if not isinstance(data, dict):
                        data = {}
                    summary = str(data.get("summary", "") or "")
                    final_conclusion = str(data.get("final_conclusion", "") or "")
                    content = ReportContent(summary=summary, final_conclusion=final_conclusion)
                except Exception:
                    content = None

            # critique_points はLLM任せにせず、入力から決定的に構成する（タグ/文字数/重複の安定化）
            points: list[str] = []

            def add_points(tag: str, items: list[str], limit: int) -> None:
                for x in (items or [])[:limit]:
                    s = ("" if x is None else str(x)).strip()
                    # 表示用に改行/連続空白を潰す
                    s = re.sub(r"\s+", " ", s).strip()
                    if not s:
                        continue
                    # 200文字制限
                    max_chars = 200 - (len(tag) + 4)  # "[X] " 分をざっくり差し引く
                    if max_chars < 50:
                        max_chars = 150
                    if len(s) > max_chars:
                        s = s[:max_chars].rstrip() + "…"
                    points.append(f"[{tag}] {s}".strip())

            add_points("Factual", list(getattr(critique, "factual_errors", []) or []), 4)
            add_points("Bias", list(getattr(critique, "bias_points", []) or []), 4)
            add_points("Rebuttal", list(getattr(optimistic_rebuttal, "counter_points", []) or []), 2)
            add_points("Rebuttal", list(getattr(pessimistic_rebuttal, "counter_points", []) or []), 2)

            mismatch_lines = [
                ln.strip("- ").strip()
                for ln in (self._evidence_mismatch_notes(article_text, optimistic_argument, pessimistic_argument) or "").splitlines()
                if ln.strip() and ln.strip() != "（なし）"
            ]
            has_mismatch = bool(mismatch_lines)
            add_points("EvidenceCheck", mismatch_lines, 4)

            # 重複除去（タグ込みで一意化）
            seen: set[str] = set()
            deduped: list[str] = []
            for p in points:
                key = re.sub(r"\\s+", " ", p).strip()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(p)

            critique_points = deduped[:12]

            # summary / final_conclusion を取り出し（失敗時はテンプレ合成）
            if content is not None:
                summary = (content.summary or "").strip()
                final_conclusion = (content.final_conclusion or "").strip()
            else:
                # テンプレ: 抽出事実+批評の要点で最小限のレポートを作る（案R1）
                top_facts = extracted_facts[:3] if extracted_facts else quote_lines[:3]
                facts_inline = " / ".join([x[:80] + ("…" if len(x) > 80 else "") for x in top_facts]) if top_facts else "（本文から具体情報を抽出できませんでした）"
                summary = f"この記事は、次の点が本文から読み取れます: {facts_inline}"
                final_conclusion = (
                    "抽出できた事実を踏まえると、機会（政策・対応の前進/効果）とリスク（副作用・不確実性）の両面を分けて評価する必要があります。"
                )

            # final_conclusion 末尾の不要記号を軽く正規化（モデルによって引用符が混入することがある）
            final_conclusion = re.sub(r'[\"”]+\\}?\\s*$', "", final_conclusion).strip()

            # --- Phase4 品質ガード（一般論/根拠なし断定の抑制） ---
            anchors = []
            anchors.extend([ln for ln in quote_lines[:6] if ln])
            anchors.extend([f for f in extracted_facts[:8] if f])

            # summary が本文アンカーに寄っていなければ、テンプレ要約に寄せる
            if self._grounding_score(summary, anchors) < 2:
                summary = self._synthesize_summary_from_facts(extracted_facts, quote_lines)

            # conclusion が弱い/一般論すぎる場合は、テンプレ結論に寄せる
            if self._grounding_score(final_conclusion, anchors) < 2:
                final_conclusion = self._synthesize_conclusion_from_facts(
                    extracted_facts=extracted_facts,
                    unknowns=unknowns,
                    critique_points=critique_points,
                    quote_lines=quote_lines,
                    has_mismatch=has_mismatch,
                )

            # summaryが抽象的すぎる場合は、本文引用候補を使って最低限の具体性を付与する
            if quote_lines:
                # 具体情報が少ない場合（数字が無い/引用断片が入っていない/抽象語が多い）に追記する
                genericish = any(tok in summary for tok in ["一般的に", "重要", "必要", "求められる", "注目", "議論", "影響"])
                lacks_quote_anchor = all((q[:20] not in summary) for q in quote_lines[:2])
                if (not re.search(r"\d", summary)) and lacks_quote_anchor and genericish:
                    q1 = quote_lines[0]
                    q2 = quote_lines[1] if len(quote_lines) > 1 else ""
                    q1 = q1[:80] + ("…" if len(q1) > 80 else "")
                    q2 = q2[:80] + ("…" if len(q2) > 80 else "")
                    extra = f"（本文より: {q1}"
                    if q2:
                        extra += f" / {q2}"
                    extra += "）"
                    # 長文化しすぎないように末尾に短く付与
                    summary = (summary + " " + extra).strip()

            # final_conclusionの必須フレーズを強制（モデルが守らないケース対策）
            if "確実度が高い点" not in final_conclusion or "不確かな点" not in final_conclusion:
                q = quote_lines[0][:80] + ("…" if quote_lines and len(quote_lines[0]) > 80 else "") if quote_lines else ""
                hi = f"本文抜粋（「{q}」）に基づく範囲の事実。".strip() if q else "本文から直接確認できる範囲の事実。"
                lo = "アナリストの引用の一部は本文一致しない可能性があり、追加検証が必要。 " if has_mismatch else "記事本文だけでは影響評価や因果の断定が難しい点。 "
                final_conclusion = (final_conclusion + f" 確実度が高い点: {hi} 不確かな点: {lo}").strip()

            return FinalReport(
                article_info=f"タイトル: {title}\nソース: {url}\n要約: {summary if summary else '（不明）'}",
                optimistic_view=optimistic_argument or Argument(conclusion="", evidence=[]),
                pessimistic_view=pessimistic_argument or Argument(conclusion="", evidence=[]),
                critique_points=critique_points,
                final_conclusion=final_conclusion,
            )
        except Exception as e:
            logging.getLogger(__name__).exception("レポート生成エラー: %s", e)
            critique_points: list[str] = []
            try:
                critique_points.extend(list(getattr(critique, "bias_points", []) or []))
                critique_points.extend(list(getattr(critique, "factual_errors", []) or []))
            except Exception:
                critique_points = []

            return FinalReport(
                article_info="",
                optimistic_view=optimistic_argument or Argument(conclusion="", evidence=[]),
                pessimistic_view=pessimistic_argument or Argument(conclusion="", evidence=[]),
                critique_points=critique_points[:10],
                final_conclusion=f"最終レポート生成に失敗しました: {str(e)}",
            )


class ReportContent(BaseModel):
    summary: str = Field(description="記事内容の要約")
    final_conclusion: str = Field(description="統合結論")


class ExtractedFacts(BaseModel):
    key_facts: list[str] = Field(default_factory=list, description="本文から抽出した事実")
    unknowns: list[str] = Field(default_factory=list, description="本文から断定できない点")
