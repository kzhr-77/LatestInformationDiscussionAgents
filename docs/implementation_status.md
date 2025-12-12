# 現在の実装状況

**最終更新日**: 2025-12-12
**プロジェクト**: LatestInformationDiscussionAgents

本書は、現在の実装状況をまとめたものです。

## 1. フォルダ構成

プロジェクトの構造は以下のように初期化されています：

```
LatestInformationDiscussionAgents/
├── docs/                   # ドキュメント (本フォルダ)
├── src/
│   ├── agents/             # エージェントロジックの実装
│   ├── core/               # オーケストレーションロジック (LangGraph)
│   ├── models/             # データスキーマ (Pydantic)
│   ├── ui/                 # ユーザーインターフェース (Streamlit)
│   └── utils/              # ユーティリティ (LLM, ツール)
├── main.py                 # CLIエントリーポイント
├── requirements.txt        # 依存ライブラリ
└── .env                    # 設定 (APIキー)
```

## 2. コンポーネントの状態

### エージェント (`src/agents/`)
| エージェントファイル | 役割 | 現在の状態 | 未実装のロジック |
| :--- | :--- | :--- | :--- |
| `researcher.py` | ニュース記事取得 (フェーズ 0) | ✅ **実装完了** | - |
| | | - URL判定機能 | |
| | | - URLからの記事取得（BeautifulSoup） | |
| | | - Tavily検索統合 | |
| | | - エラーハンドリング | |
| `analyst_optimistic.py` | 楽観的分析 (フェーズ 1, 3) | ✅ **フェーズ1実装完了** | フェーズ3（反論）の実装 |
| | | - LLMプロンプト実装 | |
| | | - 構造化出力（Argument型） | |
| | | - エラーハンドリング | |
| `analyst_pessimistic.py` | 悲観的分析 (フェーズ 1, 3) | ✅ **フェーズ1実装完了** | フェーズ3（反論）の実装 |
| | | - LLMプロンプト実装 | |
| | | - 構造化出力（Argument型） | |
| | | - エラーハンドリング | |
| `fact_checker.py` | ファクトチェック (フェーズ 2) | クラス定義済み。`validate` メソッドはモック文字列を返す。 | バイアス検知と事実検証のためのLLMプロンプト。 |
| `reporter.py` | 最終レポート (フェーズ 4) | クラス定義済み。`create_report` メソッドはモック文字列を返す。 | 統合ロジック、レポート整形プロンプト。 |

### コアロジック (`src/core/`)
- **`state.py`**:
    - `DiscussionState` (TypedDict) が定義され、主張、批評、最終レポートのフィールドを持つ。
    - ✅ `optimistic_rebuttal` と `pessimistic_rebuttal` フィールドを追加済み。
- **`graph.py`**:
    - `StateGraph` が定義され、Researcher -> Analysts -> Checker -> Reporter -> End と接続されている。
    - **状態**: フローは直線的で簡素化されている。「討論」フェーズ (フェーズ 3) のループや、並列実行（LangGraphは分岐を処理するが）の厳密な定義はまだない。

### モデル (`src/models/`)
- **`schemas.py`**: 基本的なPydanticモデル (`Argument`, `Critique`, `FinalReport`, `Rebuttal`) が定義されている。

### ユーザーインターフェース (`src/ui/`)
- **`streamlit_app.py`**:
    - 基本的なUIが実装されている。
    - 入力: トピック/URL、APIキー。
    - 出力: グラフからのモック辞書出力を表示。
    - **状態**: モック化されたグラフに接続済み。

## 3. 実装待ちタスク

`仕様.md` に基づいてシステムを完成させるためには、以下のロジックを実装する必要があります：

1.  ✅ **検索統合**: `ResearcherAgent` に `TavilySearchResults` を実装する。**完了**
2.  ✅ **LLMプロンプト（楽観的・悲観的アナリスト）**: フェーズ1用のプロンプトを実装。**完了**
3.  ✅ **構造化出力（楽観的・悲観的アナリスト）**: `.with_structured_output()` を使用して `Argument` 型を返すように実装。**完了**
4.  **討論ロジック**:
    - ✅ `DiscussionState` を更新し、反論データを保持できるようにする。**完了**
    - `graph.py` を更新し、アナリストが批評に応答する「討論」ステップ (フェーズ 3) を含める。
5.  **グラフの洗練**: ステップ間でデータが正しく渡されるようにする（例：記事をアナリストへ、主張をチェッカーへ）。

