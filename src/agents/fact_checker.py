from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from src.models.schemas import Argument, Critique

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
                "article_text": article_text[:5000],  # 長すぎる場合は切り詰め
                "optimistic_conclusion": optimistic_argument.conclusion,
                "optimistic_evidence": optimistic_evidence_str if optimistic_evidence_str else "（証拠なし）",
                "pessimistic_conclusion": pessimistic_argument.conclusion,
                "pessimistic_evidence": pessimistic_evidence_str if pessimistic_evidence_str else "（証拠なし）"
            })
            
            return result
            
        except Exception as e:
            # エラーが発生した場合、フォールバックとしてモックデータを返す
            print(f"ファクトチェックエラー: {e}")
            return Critique(
                bias_points=[f"検証中にエラーが発生しました: {str(e)}"],
                factual_errors=[]
            )

