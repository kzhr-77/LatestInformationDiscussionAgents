# エージェント設計書

**作成日**: 2025-12-07
**プロジェクト**: LatestInformationDiscussionAgents

本書は、仕様書に基づいた各エージェントの詳細設計をまとめたものです。

---

## 1. リサーチャーエージェント (ResearcherAgent)

### 1.1 役割と責任
- **フェーズ**: フェーズ 0
- **目的**: 外部から分析対象のニュース記事テキストを取得し、後続のエージェントに提供する
- **視点**: 中立的な情報収集

### 1.2 入力・出力仕様
- **入力**:
  - `topic: str` - 検索キーワードまたはURL
- **出力**:
  - `ArticleText: str` - 取得した記事の全文テキスト
  - （将来的に拡張）`ArticleMetadata: dict` - タイトル、ソース、日付など

### 1.3 実装メソッド

#### `run(topic: str) -> str`
記事を取得するメイン処理。

**処理フロー**:
1. 入力がURLかキーワードかを判定
2. URLの場合: 直接コンテンツを取得（BeautifulSoup等を使用）
3. キーワードの場合: **RSS/公式フィード許可リスト方式**で記事候補を検索（安全重視・外部検索API不要）
4. （任意）Tavily検索APIは **APIキーがある場合のみフォールバック**として利用
5. 記事テキストを抽出・整形
6. テキストを返却

**使用ツール**:
- RSS/Atom取得・パース（許可リスト: `RSS_FEED_URLS` または `config/rss_feeds.txt`）
- `TavilySearchResults`（キーワード検索の任意フォールバック: `TAVILY_API_KEY` がある場合のみ）
- ウェブスクレイピングライブラリ（URL直接取得用）
- （実装）SSRF対策ユーティリティ（`src/utils/security.py`）でURL検証・サイズ上限・リダイレクト制御を適用

**エラーハンドリング**:
- RSSフィード内にキーワード一致が見つからない場合: `RssKeywordNotFoundError` を送出（上位のオーケストレーションで通知して早期終了）
- 検索結果が見つからない/取得失敗の場合: `ValueError` を送出（上位でログ＋フォールバック）
- ネットワークエラー: 例外を捕捉し、上位でフォールバック/エラーメッセージ化（リトライは未実装）

**セキュリティ（実装済み）**:
- URL直入力は `ALLOW_URL_FETCH=0` で無効化可能
- RSS許可リストは `RSS_FEEDS_FILE_ONLY=1`（推奨）でファイル運用に固定可能
- URL検証（スキーム制限、userinfo禁止、DNS解決+拒否IPレンジ、ドメイン許可リスト、リダイレクト制御、サイズ上限）

---

## 2. 楽観的アナリストエージェント (OptimisticAnalystAgent)

### 2.1 役割と責任
- **フェーズ**: フェーズ 1（一次主張）、フェーズ 3（反論）
- **目的**: 記事からメリット、成長、機会などの前向きな要素を抽出し、主張を提示する
- **視点**: ポジティブ解釈、機会の最大化

### 2.2 入力・出力仕様

#### フェーズ 1: `analyze(article_text: str) -> Argument`
- **入力**:
  - `article_text: str` - 分析対象の記事テキスト
- **出力**:
  - `Argument` (Pydanticモデル)
    - `conclusion: str` - 楽観的な結論（1-2文）
    - `evidence: List[str]` - 記事からの具体的な引用・証拠（3-5個）

#### フェーズ 3: `debate(critique: Critique, opponent_argument: Argument) -> Rebuttal`
- **入力**:
  - `critique: Critique` - ファクトチェッカーからの批判
  - `opponent_argument: Argument` - 相手（悲観的アナリスト）の主張
  - （実装）`original_argument: Argument` - 自分の主張（フェーズ1の出力）
  - （実装）`article_text: Optional[str]` - 元記事（既定はプロンプトに入れず、環境変数で切替可能）
- **出力**:
  - `Rebuttal` (新規Pydanticモデル)
    - `counter_points: List[str]` - 相手の主張への反論ポイント
    - `strengthened_evidence: List[str]` - 自分の主張を補強する追加証拠

### 2.3 プロンプト設計方針

#### フェーズ 1用プロンプト
```
あなたは楽観的アナリストです。以下の記事を読み、以下の視点から分析してください：

1. **機会とメリット**: この記事が示す成長機会、ポジティブな影響、メリットを特定する
2. **証拠の抽出**: 記事から具体的な引用（数値、事実、引用文）を抽出し、あなたの結論を裏付ける
3. **前向きな解釈**: 一見ネガティブに見える情報も、長期的な視点でポジティブに解釈する

記事:
{article_text}

出力形式:
- 結論: [1-2文で楽観的な結論を述べる]
- 証拠: [記事からの具体的な引用を3-5個リストアップ]
```

#### フェーズ 3用プロンプト
```
あなたは楽観的アナリストです。ファクトチェッカーからの批判と、悲観的アナリストの主張を受け取りました。

あなたのタスク:
1. 悲観的アナリストの主張の弱点や矛盾点を指摘する
2. ファクトチェッカーの批判に対して、自分の主張を補強する証拠を提示する
3. 記事の文脈を再確認し、自分の解釈が正しいことを示す

あなたの元の主張:
{original_argument}

悲観的アナリストの主張:
{opponent_argument}

ファクトチェッカーの批判:
{critique}

反論を生成してください。
```

### 2.4 実装上の注意点
- LLMの構造化出力機能（`.with_structured_output()`）を使用して `Argument` モデルに直接マッピング
- 温度パラメータ: `temperature=0.7`（創造性と一貫性のバランス）
- （実装）プロンプト文字列の組み立てはヘルパーで行い、`evidence=None` 等でも落ちないように防御している
- （実装）`ENABLE_REBUTTAL_ARTICLE_CONTEXT=1` のときのみ、反論プロンプトへ `article_text` を追加する

---

## 3. 悲観的アナリストエージェント (PessimisticAnalystAgent)

### 3.1 役割と責任
- **フェーズ**: フェーズ 1（一次主張）、フェーズ 3（反論）
- **目的**: 記事からリスク、コスト、課題などの否定的な要素を抽出し、主張を提示する
- **視点**: ネガティブ解釈、リスクの特定

### 3.2 入力・出力仕様
楽観的アナリストと同様の構造だが、視点が逆。

#### フェーズ 1: `analyze(article_text: str) -> Argument`
- **入力**: `article_text: str`
- **出力**: `Argument` (悲観的な結論と証拠)

#### フェーズ 3: `debate(critique: Critique, opponent_argument: Argument) -> Rebuttal`
- **入力**: 批判と相手（楽観的アナリスト）の主張
- （実装）`original_argument: Argument` / `article_text: Optional[str]` を追加（楽観側と同様）
- **出力**: `Rebuttal`

### 3.3 プロンプト設計方針

#### フェーズ 1用プロンプト
```
あなたは悲観的アナリストです。以下の記事を読み、以下の視点から分析してください：

1. **リスクと課題**: この記事が示す潜在的なリスク、コスト、課題を特定する
2. **証拠の抽出**: 記事から具体的な引用（数値、事実、引用文）を抽出し、あなたの結論を裏付ける
3. **慎重な解釈**: 一見ポジティブに見える情報も、潜在的な問題や長期的なリスクの観点から解釈する

記事:
{article_text}

出力形式:
- 結論: [1-2文で悲観的な結論を述べる]
- 証拠: [記事からの具体的な引用を3-5個リストアップ]
```

#### フェーズ 3用プロンプト
楽観的アナリストと同様の構造だが、「悲観的アナリスト」として反論を生成。

### 3.4 実装上の注意点
- 楽観的アナリストと同じ実装パターンだが、プロンプトの視点が異なる
- 温度パラメータ: `temperature=0.7`

---

## 4. ファクトチェッカーエージェント (FactCheckerAgent)

### 4.1 役割と責任
- **フェーズ**: フェーズ 2
- **目的**: 楽観的・悲観的アナリストの主張を検証し、事実の正確性やバイアスを指摘する
- **視点**: 中立的、客観的、事実ベース

### 4.2 入力・出力仕様

#### `validate(optimistic_argument: Argument, pessimistic_argument: Argument, article_text: str) -> Critique`
- **入力**:
  - `optimistic_argument: Argument` - 楽観的アナリストの主張
  - `pessimistic_argument: Argument` - 悲観的アナリストの主張
  - `article_text: str` - 元の記事テキスト（検証の参照用）
- **出力**:
  - `Critique` (Pydanticモデル)
    - `bias_points: List[str]` - 各主張におけるバイアスや偏りの指摘（各アナリストごとに分けて記述）
    - `factual_errors: List[str]` - 事実の誤りや文脈からの逸脱の指摘

### 4.3 プロンプト設計方針

```
あなたは客観的なファクトチェッカーです。楽観的アナリストと悲観的アナリストの主張を検証してください。

検証ポイント:
1. **引用の正確性**: 各アナリストが引用した部分が、元の記事の文脈に合っているか
2. **誇張の検出**: 記事の内容を過度に強調したり、歪曲していないか
3. **バイアスの特定**: 各アナリストが特定の視点に偏っていないか
4. **事実の確認**: 数値やデータが正確に引用されているか

元の記事:
{article_text}

楽観的アナリストの主張:
{optimistic_argument}

悲観的アナリストの主張:
{pessimistic_argument}

検証結果を以下の形式で出力してください:
- バイアス指摘: [各アナリストの主張における偏りやバイアスを指摘]
- 事実誤り: [事実の誤りや文脈からの逸脱を指摘]
```

### 4.4 実装上の注意点
- 温度パラメータ: `temperature=0.3`（事実検証のため、低めに設定）
- （実装）structured output の失敗に備え、JSON文字列出力→パース復元を標準化（`bias_points`/`factual_errors` を必ず返す）
- （実装）JSON抽出は括弧カウントで頑健化し、失敗時は安全な短い断片のみログに残す

---

## 5. レポートエージェント (ReporterAgent)

### 5.1 役割と責任
- **フェーズ**: フェーズ 4
- **目的**: 全フェーズの出力（主張、批判、反論）を統合し、バランスの取れた最終レポートを作成する
- **視点**: 統合的、多角的、中立的

### 5.2 入力・出力仕様

#### （設計）`create_report(...全フェーズの出力...) -> FinalReport`
- **状態**: ✅ **実装済み**（`src/agents/reporter.py`）。
- **実装上の入力/出力**:
  - **入力**: `article_text`, `optimistic_argument`, `pessimistic_argument`, `critique`, `optimistic_rebuttal`, `pessimistic_rebuttal`, `article_url`
  - **出力**: `FinalReport` (Pydanticモデル)
    - `article_info: str` - 記事タイトル、ソース、要約（3行）
    - `optimistic_view: Argument` / `pessimistic_view: Argument` - stateの値をそのまま採用（幻覚混入抑制）
    - `critique_points: List[str]` - Critique/反論/本文一致チェックからタグ付きで決定的に生成
    - `final_conclusion: str` - 機会/リスク＋「確実度が高い点」「不確かな点」を含む

**実装詳細（要点）**:
- 2段構成: facts抽出 → 統合レポート生成
- structured output が不安定なモデルでも完走できるよう、JSON文字列フォールバック＋テンプレ/機械抽出フォールバックを備える

### 5.3 プロンプト設計方針

```
あなたはレポートエージェントです。全フェーズの議論を統合し、バランスの取れた最終レポートを作成してください。

レポートの構成:
1. **分析対象記事情報**: 記事タイトル、ソース、要約を抽出
2. **一次主張の要約**: 楽観的・悲観的アナリストの結論と主要論拠を整理
3. **議論の過程**: ファクトチェッカーの指摘事項と、相互反論の要点をまとめる
4. **最終統合結論**: 
   - 最も確実な事実（ファクトチェッカーの検証に基づく）
   - 潜在的な機会（メリット）とリスク（懸念）をバランスよくまとめる
   - 最も説得力があった議論のポイント

元の記事:
{article_text}

楽観的アナリストの主張:
{optimistic_argument}

悲観的アナリストの主張:
{pessimistic_argument}

ファクトチェッカーの批判:
{critique}

楽観的アナリストの反論:
{optimistic_rebuttal}

悲観的アナリストの反論:
{pessimistic_rebuttal}

最終レポートを生成してください。
```

### 5.4 実装上の注意点
- 温度パラメータ: `temperature=0.5`（統合的な結論のため、中程度）
- 構造化出力で `FinalReport` モデルに直接マッピング
- 記事情報の抽出は、リサーチャーエージェントから取得したメタデータを活用

---

## 6. データモデルの拡張

### 6.1 新規追加が必要なモデル

#### `Rebuttal` (フェーズ 3用)
```python
class Rebuttal(BaseModel):
    counter_points: List[str] = Field(description="相手の主張への反論ポイント")
    strengthened_evidence: List[str] = Field(description="自分の主張を補強する追加証拠")
```

#### `ArticleMetadata` (リサーチャーエージェント用、オプション)
```python
class ArticleMetadata(BaseModel):
    title: str
    source: str
    published_date: Optional[str]
    url: Optional[str]
```

### 6.2 `DiscussionState` の更新
```python
class DiscussionState(TypedDict, total=False):
    topic: str
    request_id: str
    halt: bool
    halt_reason: str
    article_text: str
    optimistic_argument: Optional[Argument]
    pessimistic_argument: Optional[Argument]
    critique: Optional[Critique]
    optimistic_rebuttal: Optional[Rebuttal]  # 追加
    pessimistic_rebuttal: Optional[Rebuttal]  # 追加
    final_report: Optional[FinalReport]
    messages: List[str]
```

---

## 7. 実装の優先順位

### フェーズ 1: 基本機能の実装
1. リサーチャーエージェント（Tavily統合）
2. 楽観的・悲観的アナリスト（フェーズ 1のみ）
3. ファクトチェッカー
4. レポートエージェント（簡易版）

### フェーズ 2: 討論機能の追加
5. 反論機能（フェーズ 3）の実装
6. `Rebuttal` モデルの追加と `DiscussionState` の更新
7. グラフの更新（フェーズ 3のループ追加）

### フェーズ 3: 品質向上
8. エラーハンドリングの強化
9. プロンプトの最適化
10. メタデータの活用

---

## 8. テスト戦略

各エージェントに対して以下をテスト:
- **単体テスト**: 各メソッドの入力・出力の検証
- **統合テスト**: オーケストレーションフロー全体の動作確認（現行は `OrchestrationAgent`）
- **品質テスト**: 生成される主張・レポートの品質評価（仕様書の評価ポイントに基づく）

（実装済み・軽量スモーク）
- `tools/smoke_phase3_no_ollama.py`: 外部LLM無しでフェーズ3まで完走し、出力キーが揃うことを確認
- `tools/smoke_rss_no_keyword_exits.py`: RSSキーワード一致なしの場合に通知して早期終了することを確認

---

**次のステップ**: この設計書に基づいて、各エージェントの実装を開始します。

