import streamlit as st
import os
import sys
from pathlib import Path
import uuid
import logging

# Adjust path to import src
# Note: Streamlitアプリは実行時に異なるディレクトリから起動される可能性があるため、
# プロジェクトルートを明示的に追加する必要があります
# より良い方法: PYTHONPATH環境変数を設定するか、setup.pyを使用する
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.graph import create_graph
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Discussion News Analysis", layout="wide")

st.title("討論型ニュース分析システム")

# OpenAI用（コメントアウト）
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key

# Ollama用: モデル選択（オプション）
model_name = st.sidebar.selectbox(
    "使用するモデル",
    ["gemma3:4b", "llama3:8b", "mistral:7b"],
    index=0
)

topic = st.text_input("分析したいトピックまたはURLを入力してください")

if st.button("分析開始"):
    if not topic:
        st.warning("トピックを入力してください。")
    else:
        st.info("分析を開始します...")
        
        try:
            # グラフの作成
            graph = create_graph(model_name)
            
            # 初期状態の設定
            request_id = str(uuid.uuid4())
            initial_state = {"topic": topic, "messages": [], "request_id": request_id}
            logger.info("[%s] UI開始 topic=%s model=%s", request_id, topic, model_name)
            
            # グラフの実行
            with st.spinner("分析中..."):
                result = graph.invoke(initial_state)
            
            st.success("分析完了！")

            if result.get("halt"):
                st.warning(result.get("halt_reason") or "処理を終了しました。")
                st.stop()
            
            # 結果の表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("楽観的視点")
                optimistic_arg = result.get("optimistic_argument")
                if optimistic_arg:
                    if hasattr(optimistic_arg, 'conclusion'):
                        st.write(f"**結論**: {optimistic_arg.conclusion}")
                        if optimistic_arg.evidence:
                            st.write("**証拠**:")
                            for evidence in optimistic_arg.evidence:
                                st.write(f"- {evidence}")
                    else:
                        st.write(optimistic_arg)
                else:
                    st.info("データがありません")
                
            with col2:
                st.subheader("悲観的視点")
                pessimistic_arg = result.get("pessimistic_argument")
                if pessimistic_arg:
                    if hasattr(pessimistic_arg, 'conclusion'):
                        st.write(f"**結論**: {pessimistic_arg.conclusion}")
                        if pessimistic_arg.evidence:
                            st.write("**証拠**:")
                            for evidence in pessimistic_arg.evidence:
                                st.write(f"- {evidence}")
                    else:
                        st.write(pessimistic_arg)
                else:
                    st.info("データがありません")
            
            st.subheader("ファクトチェック・批評")
            critique = result.get("critique")
            if critique:
                if hasattr(critique, 'bias_points'):
                    if critique.bias_points:
                        st.write("**バイアス指摘**:")
                        for point in critique.bias_points:
                            st.write(f"- {point}")
                    if critique.factual_errors:
                        st.write("**事実誤り**:")
                        for error in critique.factual_errors:
                            st.write(f"- {error}")
                else:
                    st.write(critique)
            else:
                st.info("データがありません")

            st.subheader("討論（反論）")
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**楽観的アナリストの反論**")
                optimistic_rebuttal = result.get("optimistic_rebuttal")
                if optimistic_rebuttal and hasattr(optimistic_rebuttal, "counter_points"):
                    if optimistic_rebuttal.counter_points:
                        st.write("**反論ポイント**:")
                        for p in optimistic_rebuttal.counter_points:
                            st.write(f"- {p}")
                    if optimistic_rebuttal.strengthened_evidence:
                        st.write("**補強証拠**:")
                        for ev in optimistic_rebuttal.strengthened_evidence:
                            st.write(f"- {ev}")
                elif optimistic_rebuttal:
                    st.write(optimistic_rebuttal)
                else:
                    st.info("データがありません")

            with col4:
                st.markdown("**悲観的アナリストの反論**")
                pessimistic_rebuttal = result.get("pessimistic_rebuttal")
                if pessimistic_rebuttal and hasattr(pessimistic_rebuttal, "counter_points"):
                    if pessimistic_rebuttal.counter_points:
                        st.write("**反論ポイント**:")
                        for p in pessimistic_rebuttal.counter_points:
                            st.write(f"- {p}")
                    if pessimistic_rebuttal.strengthened_evidence:
                        st.write("**補強証拠**:")
                        for ev in pessimistic_rebuttal.strengthened_evidence:
                            st.write(f"- {ev}")
                elif pessimistic_rebuttal:
                    st.write(pessimistic_rebuttal)
                else:
                    st.info("データがありません")
            
            st.header("最終レポート")
            final_report = result.get("final_report")
            if final_report:
                if hasattr(final_report, 'final_conclusion'):
                    st.write(f"**最終結論**: {final_report.final_conclusion}")
                    if final_report.critique_points:
                        st.write("**批評ポイント**:")
                        for point in final_report.critique_points:
                            st.write(f"- {point}")
                else:
                    st.write(final_report)
            else:
                st.info("データがありません")
            
        except ValueError as e:
            st.error(f"**設定エラー**: {e}")
            st.info("💡 **対処方法**:\n"
                   "- モデルがダウンロードされているか確認: `ollama list`\n"
                   "- モデルをダウンロード: `ollama pull {model_name}`")
        except ConnectionError as e:
            st.error(f"**接続エラー**: {e}")
            st.info("💡 **対処方法**:\n"
                   "- Ollamaサービスが起動しているか確認\n"
                   "- ターミナルで `ollama serve` を実行するか、Ollamaアプリを起動\n"
                   "- ファイアウォールがブロックしていないか確認")
        except Exception as e:
            st.error(f"**予期しないエラーが発生しました**: {e}")
            with st.expander("詳細なエラー情報"):
                st.exception(e)  # 詳細なエラー情報を表示

