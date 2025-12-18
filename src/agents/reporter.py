from __future__ import annotations

import logging
import re
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
        self.report_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """あなたはレポートエージェントです。以下の情報を統合し、バイアスの少ない多角的な最終結論と要約を作成してください。

重要ルール:
- 出力は**必ず日本語**にする
- 記事本文に無い事実を作らない（不明な点は「不明」と書く）
- ファクトチェッカーの指摘（bias/factual）を優先し、誇張を避ける
- 引用（evidence）が記事本文に見当たらない場合は「引用根拠が本文に見当たらない可能性」を明示する
- 結論は「機会」と「リスク」を両方扱い、単なる折衷案にならないように要点を整理する
- 一般論（「多角的に検討が必要」「慎重な議論が重要」等）だけで終わらせない。本文の具体情報に触れること。
- 記事内容の根拠として、下の「本文からの引用候補（抜粋）」を優先して参照する（そこに無い内容は断定しない）。

出力は次の構造（ReportContent）に合わせること:
- summary: 記事内容の要約（2〜5文）。**少なくとも2つ**は具体情報（数字/固有名詞/出来事/決定事項）に触れる。
- final_conclusion: 議論を踏まえた統合結論（2〜6文）。最後に必ず「確実度が高い点: ...」「不確かな点: ...」を1文ずつ含める。""",
                ),
                (
                    "human",
                    """記事タイトル:
{article_title}

ソースURL:
{article_url}

元の記事（抜粋）:
{article_text}

本文からの引用候補（抜粋）:
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
        lines = [re.sub(r"\s+", " ", (ln or "")).strip() for ln in (article_body or "").splitlines()]
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
            chain = self.report_prompt | self.model.with_structured_output(ReportContent)
            content: ReportContent = chain.invoke(
                {
                    "article_title": title,
                    "article_url": url,
                    "article_text": self._truncate(body, 8000),
                    "article_quotes": self._pick_article_quotes(body, limit=6),
                    "optimistic_argument": self._fmt_argument(optimistic_argument),
                    "pessimistic_argument": self._fmt_argument(pessimistic_argument),
                    "critique": self._fmt_critique(critique),
                    "optimistic_rebuttal": self._fmt_rebuttal(optimistic_rebuttal),
                    "pessimistic_rebuttal": self._fmt_rebuttal(pessimistic_rebuttal),
                    "evidence_mismatch_notes": self._evidence_mismatch_notes(article_text, optimistic_argument, pessimistic_argument),
                }
            )

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

            # final_conclusion 末尾の不要記号を軽く正規化（モデルによって引用符が混入することがある）
            final_conclusion = (content.final_conclusion or "").strip()
            final_conclusion = re.sub(r'[\"”]+\\}?\\s*$', "", final_conclusion).strip()

            # summaryが抽象的すぎる場合は、本文引用候補を使って最低限の具体性を付与する
            summary = (content.summary or "").strip()
            quote_lines = [ln.strip()[2:].strip() for ln in self._pick_article_quotes(body, limit=4).splitlines() if ln.strip().startswith("- ")]
            if quote_lines:
                # 具体情報が少ない場合（数字/固有名詞/引用符が無い）に追記する
                if (not re.search(r"\d", summary)) and (all(q[:20] not in summary for q in quote_lines[:2])):
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
