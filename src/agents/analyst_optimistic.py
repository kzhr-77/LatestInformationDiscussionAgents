from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from src.models.schemas import Argument, Critique, Rebuttal
import logging
import os

class OptimisticAnalystAgent:
    """
    楽観的アナリストエージェント
    
    記事からメリット、成長、機会などの前向きな要素を抽出し、主張を提示する。
    """
    
    def __init__(self, model: BaseChatModel):
        """
        楽観的アナリストエージェントを初期化
        
        Args:
            model: LLMモデル
        """
        self.model = model
        self._init_prompts()
    
    def _init_prompts(self):
        """プロンプトテンプレートを初期化"""
        # フェーズ1用プロンプト
        self.analyze_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは楽観的アナリストです。以下の記事を読み、以下の視点から分析してください：

1. **機会とメリット**: この記事が示す成長機会、ポジティブな影響、メリットを特定する
2. **証拠の抽出**: 記事から具体的な引用（数値、事実、引用文）を抽出し、あなたの結論を裏付ける
3. **前向きな解釈**: 一見ネガティブに見える情報も、長期的な視点でポジティブに解釈する

出力は以下の形式で構造化してください：
- conclusion: 1-2文で楽観的な結論を述べる
- evidence: 記事からの具体的な引用を3-5個リストアップ（各引用は記事の文脈を保った形で）"""),
            ("human", "記事:\n{article_text}")
        ])
        
        # フェーズ3用プロンプト
        # - 既定は「修正前と挙動（LLM入力）を変えない」ため、記事本文は渡さない版を使う
        # - ENABLE_REBUTTAL_ARTICLE_CONTEXT=1 のときのみ、記事本文も与える版を使う
        self.debate_prompt_basic = ChatPromptTemplate.from_messages([
            ("system", """あなたは楽観的アナリストです。ファクトチェッカーからの批判と、悲観的アナリストの主張を受け取りました。

あなたのタスク:
1. 悲観的アナリストの主張の弱点や矛盾点を指摘する
2. ファクトチェッカーの批判に対して、自分の主張を補強する証拠を提示する
3. 記事の文脈を再確認し、自分の解釈が正しいことを示す

出力は以下の形式で構造化してください：
- counter_points: 相手の主張への反論ポイント（2-3個）
- strengthened_evidence: 自分の主張を補強する追加証拠（2-3個）"""),
            ("human", """あなたの元の主張:
{original_argument}

悲観的アナリストの主張:
{opponent_argument}

ファクトチェッカーの批判:
{critique}

反論を生成してください。""")
        ])

        self.debate_prompt_with_article = ChatPromptTemplate.from_messages([
            ("system", """あなたは楽観的アナリストです。ファクトチェッカーからの批判と、悲観的アナリストの主張を受け取りました。

あなたのタスク:
1. 悲観的アナリストの主張の弱点や矛盾点を指摘する
2. ファクトチェッカーの批判に対して、自分の主張を補強する証拠を提示する
3. 記事の文脈を再確認し、自分の解釈が正しいことを示す

出力は以下の形式で構造化してください：
- counter_points: 相手の主張への反論ポイント（2-3個）
- strengthened_evidence: 自分の主張を補強する追加証拠（2-3個）"""),
            ("human", """あなたの元の主張:
{original_argument}

悲観的アナリストの主張:
{opponent_argument}

ファクトチェッカーの批判:
{critique}

元の記事（参考・必要なら引用）:
{article_text}

反論を生成してください。""")
        ])

    @staticmethod
    def _format_argument_for_prompt(argument: Argument) -> str:
        conclusion = "" if argument is None else str(getattr(argument, "conclusion", "") or "")
        evidence = getattr(argument, "evidence", []) if argument is not None else []
        if evidence is None:
            evidence = []
        evidence_lines = "\n".join([f"- {ev}" for ev in evidence]) if evidence else "（証拠なし）"
        return f"結論: {conclusion}\n証拠:\n{evidence_lines}"

    @staticmethod
    def _format_critique_for_prompt(critique: Critique) -> str:
        bias_points = getattr(critique, "bias_points", []) if critique is not None else []
        factual_errors = getattr(critique, "factual_errors", []) if critique is not None else []
        if bias_points is None:
            bias_points = []
        if factual_errors is None:
            factual_errors = []
        bias = "\n".join([f"- {x}" for x in bias_points]) if bias_points else "（なし）"
        factual = "\n".join([f"- {x}" for x in factual_errors]) if factual_errors else "（なし）"
        return f"バイアス指摘:\n{bias}\n事実誤り:\n{factual}"
    
    def analyze(self, article_text: str) -> Argument:
        """
        記事を楽観的な視点から分析する（フェーズ1）
        
        Args:
            article_text: 分析対象の記事テキスト
        
        Returns:
            Argument: 楽観的な結論と証拠
        
        Raises:
            ValueError: 記事テキストが空の場合
        """
        if not article_text or not article_text.strip():
            raise ValueError("記事テキストが空です。")
        
        try:
            # プロンプトチェーンを作成
            chain = self.analyze_prompt | self.model.with_structured_output(Argument)
            
            # LLMを呼び出して構造化出力を取得
            result = chain.invoke({"article_text": article_text})
            
            return result
            
        except Exception as e:
            # エラーが発生した場合、フォールバックとしてモックデータを返す
            logging.getLogger(__name__).exception("楽観的分析エラー: %s", e)
            return Argument(
                conclusion=f"分析中にエラーが発生しました: {str(e)}",
                evidence=[]
            )
    
    def debate(
        self,
        critique: Critique,
        opponent_argument: Argument,
        original_argument: Argument,
        article_text: str | None = None,
    ) -> Rebuttal:
        """
        ファクトチェッカーの批判と相手の主張に対して反論する（フェーズ3）
        
        Args:
            critique: ファクトチェッカーからの批判
            opponent_argument: 悲観的アナリストの主張
            original_argument: 自分（楽観的アナリスト）の主張（フェーズ1の出力）
            article_text: 元の記事テキスト（参考・必要なら引用）
        
        Returns:
            Rebuttal: 反論ポイントと補強証拠
        """
        try:
            # プロンプトチェーンを作成（既定は修正前と同一入力。環境変数でのみ本文コンテキストを追加）
            use_article = os.getenv("ENABLE_REBUTTAL_ARTICLE_CONTEXT", "0") == "1"
            prompt = self.debate_prompt_with_article if use_article else self.debate_prompt_basic
            chain = prompt | self.model.with_structured_output(Rebuttal)
            
            # LLMを呼び出して構造化出力を取得
            result = chain.invoke({
                "original_argument": self._format_argument_for_prompt(original_argument),
                "opponent_argument": self._format_argument_for_prompt(opponent_argument),
                "critique": self._format_critique_for_prompt(critique),
                "article_text": (article_text or "").strip(),
            })
            
            return result
            
        except Exception as e:
            # エラーが発生した場合、フォールバックとしてモックデータを返す
            logging.getLogger(__name__).exception("楽観的反論エラー: %s", e)
            return Rebuttal(
                counter_points=[f"エラー: {str(e)}"],
                strengthened_evidence=[]
            )

