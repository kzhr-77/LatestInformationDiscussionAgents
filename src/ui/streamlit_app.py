import streamlit as st
import os
import sys

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.graph import create_graph

st.set_page_config(page_title="Discussion News Analysis", layout="wide")

st.title("討論型ニュース分析システム")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

topic = st.text_input("分析したいトピックまたはURLを入力してください")

if st.button("分析開始"):
    if not topic:
        st.warning("トピックを入力してください。")
    elif not api_key and not os.getenv("OPENAI_API_KEY"):
        st.warning("API Keyを設定してください。")
    else:
        st.info("分析を開始します...")
        graph = create_graph()
        try:
            initial_state = {"topic": topic, "messages": []}
            result = graph.invoke(initial_state)
            
            st.success("分析完了！")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("楽観的視点")
                st.write(result.get("optimistic_argument"))
                
            with col2:
                st.subheader("悲観的視点")
                st.write(result.get("pessimistic_argument"))
            
            st.subheader("ファクトチェック・批評")
            st.write(result.get("critique"))
            
            st.header("最終レポート")
            st.write(result.get("final_report"))
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

