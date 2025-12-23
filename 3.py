# -*- coding: utf-8 -*-
import streamlit as st
import jieba
import requests
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# 页面基础配置
st.set_page_config(page_title="URL词频分析系统", page_icon="📊", layout="wide")

# 固定测试文本（爬取失败时兜底）
TEST_TEXT = """人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学。它涵盖了机器学习、自然语言处理、计算机视觉、专家系统等多个领域。机器学习是人工智能的核心，通过让计算机从数据中学习模式，而无需显式编程。深度学习作为机器学习的一个分支，使用神经网络模拟人脑结构，在图像识别、语音识别等领域取得了突破性进展。自然语言处理则专注于让计算机理解和生成人类语言，如聊天机器人、机器翻译等应用。人工智能的发展已经深刻影响了医疗、金融、交通、教育等各行各业，未来还将继续推动社会的数字化转型。"""

# 新增：URL爬取函数
def crawl_url_text(url):
    """爬取指定URL的中文文本内容"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        p_text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True))>5])
        art_text = soup.find("article").get_text(strip=True) if soup.find("article") else ""
        raw_text = p_text if len(p_text) > len(art_text) else art_text
        clean_text = re.sub(r"[^\u4e00-\u9fa5\s]", "", re.sub(r"\s+", " ", raw_text))
        return clean_text if len(clean_text) > 50 else None
    except Exception as e:
        st.error(f"URL爬取失败：{str(e)}")
        return None

# 1. 文本分析（生成带编号的词汇表）
def analyze_text(text, min_freq=1):
    stop_words = {'的','了','是','在','和','有','就','都','这','那','个','为','把','我','你','他','她','它','我们','你们','他们','这里','那里','什么','怎么','为什么','如何','然后','但是','如果','因为','所以','虽然','既然','之','于','也','还','及','与','或','即','所','将','会','可','能','应','该','要','需','须','得','过','着','啊','呀','呢','吗','吧'}
    words = [w for w in jieba.lcut(text) if w not in stop_words and len(w)>1]
    word_freq = Counter(words)
    top10 = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)[:10]
    df = pd.DataFrame(top10, columns=["词汇", "词频"])
    df["编号"] = [f"#{i+1}" for i in range(len(df))]
    return df

# 2. 图表生成（核心修改：去掉标题后的方框）
def render_chart(df, chart_type):
    df_chart = df.copy()
    df_chart["显示标签"] = df_chart["编号"]
    df_chart = df_chart.set_index("显示标签")

    if chart_type == "柱状图":
        st.bar_chart(df_chart["词频"], use_container_width=True)
    elif chart_type == "折线图":
        st.line_chart(df_chart["词频"], use_container_width=True)
    elif chart_type == "面积图":
        st.area_chart(df_chart["词频"], use_container_width=True)
    elif chart_type == "饼图":
        fig, ax = plt.subplots(figsize=(8,8))
        ax.pie(
            df["词频"], 
            labels=df["编号"],
            autopct='%1.1f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(df)))
        )
        ax.set_title("TOP10词汇饼图（编号对应下方中文）")  # 去掉多余方框
        st.pyplot(fig)
    elif chart_type == "散点图":
        st.scatter_chart(df, x="编号", y="词频", size="词频", use_container_width=True)
    elif chart_type == "横向柱状图（替代词云）":
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(df["编号"], df["词频"], color="#4285F4")
        ax.set_xlabel("词频")
        ax.set_ylabel("词汇编号")
        ax.set_title("TOP10词汇横向柱状图（替代词云）")  # 去掉多余方框
        st.pyplot(fig)
    elif chart_type == "热力图（数值）":
        st.dataframe(df_chart[["词频"]].style.background_gradient(cmap="Blues"), use_container_width=True)
    elif chart_type == "漏斗图（排序）":
        df_sorted = df.sort_values("词频", ascending=True)
        df_sorted = df_sorted.set_index("编号")
        st.bar_chart(df_sorted["词频"], use_container_width=True)

    # 外置中文标注
    st.markdown("### 📝 图表编号-中文词汇对应表")
    label_df = df[["编号", "词汇", "词频"]].set_index("编号")
    st.dataframe(label_df, use_container_width=True)

# ======== 页面布局 ========
st.title("📊 URL词频分析系统（最终稳定版）")
st.markdown("### 网址爬取配置")

url_input = st.text_input(
    label="输入需要爬取的网站URL",
    placeholder="示例：https://www.guokr.com/article/440923/",
    help="请输入公开的中文文章类URL（如新闻、博客、公众号文章）"
)

st.markdown("### 分析配置项")
with st.sidebar:
    min_freq = st.slider("最低词频过滤", 1, 5, 1)
    chart_type = st.selectbox(
        "选择图表类型",
        [
            "柱状图", "折线图", "面积图", "饼图", 
            "散点图", "横向柱状图（替代词云）", 
            "热力图（数值）", "漏斗图（排序）"
        ]
    )
    analyze_btn = st.button("🔍 开始爬取并分析", type="primary")

# 核心逻辑
if analyze_btn:
    if not url_input:
        st.warning("请先输入需要爬取的URL地址！")
    else:
        with st.spinner("正在爬取网页文本..."):
            crawled_text = crawl_url_text(url_input)
            target_text = crawled_text if crawled_text else TEST_TEXT
            if not crawled_text:
                st.info("爬取失败，自动使用测试文本进行分析")
        
        df_result = analyze_text(target_text, min_freq)
        st.success("✅ 分析完成！")
        st.markdown("### 📋 TOP10词汇原始列表")
        st.dataframe(df_result[["词汇", "词频"]], use_container_width=True)
        st.markdown(f"### 📈 {chart_type}")
        render_chart(df_result, chart_type)

# 页脚说明
st.divider()
st.caption("💡 图表内用编号保证样式，下方标注中文词汇，兼顾可视化效果与可读性")
