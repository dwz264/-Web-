# -*- coding: utf-8 -*-
import streamlit as st
import jieba
from collections import Counter
import pandas as pd

# 页面基础配置
st.set_page_config(page_title="URL词频分析", page_icon="📊", layout="wide")

# 固定测试文本
TEST_TEXT = """人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学。它涵盖了机器学习、自然语言处理、计算机视觉、专家系统等多个领域。机器学习是人工智能的核心，通过让计算机从数据中学习模式，而无需显式编程。深度学习作为机器学习的一个分支，使用神经网络模拟人脑结构，在图像识别、语音识别等领域取得了突破性进展。自然语言处理则专注于让计算机理解和生成人类语言，如聊天机器人、机器翻译等应用。人工智能的发展已经深刻影响了医疗、金融、交通、教育等各行各业，未来还将继续推动社会的数字化转型。"""

# 1. 文本分析
def analyze_text(text, min_freq=1):
    stop_words = {'的','了','是','在','和','有','就','都','这','那','个','为','把','我','你','他','她','它','我们','你们','他们','这里','那里','什么','怎么','为什么','如何','然后','但是','如果','因为','所以','虽然','既然','之','于','也','还','及','与','或','即','所','将','会','可','能','应','该','要','需','须','得','过','着','啊','呀','呢','吗','吧'}
    words = [w for w in jieba.lcut(text) if w not in stop_words and len(w)>1]
    word_freq = Counter(words)
    top20 = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)[:20]
    return pd.DataFrame(top20, columns=["词汇", "词频"])

# 2. 图表生成（纯Streamlit原生组件，中文100%显示）
def render_chart(df, chart_type):
    df = df.head(10)  # 限制TOP10，避免拥挤
    max_freq = df["词频"].max()

    if chart_type == "柱状图":
        st.bar_chart(df.set_index("词汇")["词频"], use_container_width=True)
    elif chart_type == "折线图":
        st.line_chart(df.set_index("词汇")["词频"], use_container_width=True)
    elif chart_type == "面积图":
        st.area_chart(df.set_index("词汇")["词频"], use_container_width=True)
    elif chart_type == "饼图":
        # 原生饼图+文字标签
        st.markdown("### TOP10词汇占比")
        for idx, (word, freq) in df.iterrows():
            st.write(f"**{word}** ({freq}次，占比{round(freq/max_freq*100,1)}%)")
            st.progress(freq/max_freq)
    elif chart_type == "散点图":
        st.scatter_chart(df, x="词汇", y="词频", size="词频", use_container_width=True)
    elif chart_type == "横向柱状图（替代词云）":
        # 原生横向柱状图+中文标签
        st.markdown("### TOP10词汇词频（横向）")
        for word, freq in zip(df["词汇"], df["词频"]):
            st.write(f"**{word}**")
            st.progress(freq/max_freq)
    elif chart_type == "热力图（数值）":
        st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)
    elif chart_type == "漏斗图（排序）":
        st.bar_chart(df.sort_values("词频", ascending=True).set_index("词汇")["词频"], use_container_width=True)

# 页面布局
st.title("📊 URL词频分析系统（中文稳定版）")
st.markdown("### 配置项")

# 侧边栏
with st.sidebar:
    min_freq = st.slider("最低词频过滤", 1, 5, 1)
    chart_type = st.selectbox(
        "选择图表类型",
        ["柱状图", "折线图", "面积图", "饼图", "散点图", "横向柱状图（替代词云）", "热力图（数值）", "漏斗图（排序）"]
    )
    analyze_btn = st.button("🔍 开始分析", type="primary")

# 核心逻辑
if analyze_btn:
    df_result = analyze_text(TEST_TEXT, min_freq)
    st.success("✅ 分析完成！")
    st.markdown("### 📋 TOP10词汇列表")
    st.dataframe(df_result.head(10), use_container_width=True)
    st.markdown(f"### 📈 {chart_type}")
    render_chart(df_result, chart_type)

# 页脚
st.divider()
st.caption("💡 纯Streamlit原生组件，中文100%显示")
