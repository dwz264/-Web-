# -*- coding: utf-8 -*-
import streamlit as st
import jieba
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup
import pandas as pd

# 页面配置
st.set_page_config(page_title="URL词频分析系统", page_icon="📊", layout="wide")

# 兜底文本
BACKUP_TEXT = """人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学。它涵盖了机器学习、自然语言处理、计算机视觉、专家系统等多个领域。机器学习是人工智能的核心，通过让计算机从数据中学习模式，而无需显式编程。深度学习作为机器学习的一个分支，使用神经网络模拟人脑结构，在图像识别、语音识别等领域取得了突破性进展。自然语言处理则专注于让计算机理解和生成人类语言，如聊天机器人、机器翻译等应用。人工智能的发展已经深刻影响了医疗、金融、交通、教育等各行各业，未来还将继续推动社会的数字化转型。"""

# 1. URL文本抓取
def fetch_url_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) Chrome/86.0.4240.198 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        resp.encoding = resp.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        p_text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True))>10])
        art_text = soup.find("article").get_text(strip=True) if soup.find("article") else ""
        text = p_text if len(p_text) > len(art_text) else art_text
        return re.sub(r'[^\u4e00-\u9fa5\s]', '', re.sub(r'\s+', ' ', text))
    except Exception as e:
        return f"URL抓取失败：{str(e)}"

# 2. 文本分析
def analyze_text(text, min_freq=1):
    stop_words = {'的','了','是','在','和','有','就','都','这','那','个','为','把','我','你','他','她','它','我们','你们','他们','这里','那里','什么','怎么','为什么','如何','然后','但是','如果','因为','所以','虽然','既然','之','于','也','还','及','与','或','即','所','将','会','可','能','应','该','要','需','须','得','过','着','啊','呀','呢','吗','吧'}
    words = [w for w in jieba.lcut(re.sub(r'\s+', ' ', text)) if w not in stop_words and len(w)>1]
    word_freq = Counter(words)
    return {k:v for k,v in word_freq.items() if v>=min_freq}, sorted(word_freq.items(), key=lambda x:x[1], reverse=True)[:20]

# 3. 纯Streamlit原生图表（8种，无任何第三方库）
def show_chart(top20, chart_type):
    if not top20:
        st.warning("暂无有效数据")
        return
    df = pd.DataFrame(top20, columns=["词汇", "词频"])

    # 1. 词云图（原生组件版）
    if chart_type == "词云图":
        st.subheader("TOP20词汇词云图")
        cols = st.columns(5)
        for idx, (word, freq) in enumerate(top20):
            col = cols[idx % 5]
            if freq >= 8:
                col.title(word)
            elif freq >= 5:
                col.header(word)
            else:
                col.subheader(word)

    # 2. 柱状图（原生表格+进度条）
    elif chart_type == "柱状图":
        st.subheader("TOP20词汇柱状图")
        for word, freq in top20:
            st.write(f"**{word}**")
            st.progress(freq / df["词频"].max())

    # 3. 折线图（原生折线组件）
    elif chart_type == "折线图":
        st.subheader("TOP20词汇折线图")
        st.line_chart(df.set_index("词汇")["词频"])

    # 4. 饼图（原生饼图组件）
    elif chart_type == "饼图":
        st.subheader("TOP20词汇饼图")
        st.pie_chart(df.set_index("词汇")["词频"])

    # 5. 雷达图（原生指标+分栏）
    elif chart_type == "雷达图":
        st.subheader("TOP8词汇雷达图")
        df_radar = df.head(8)
        cols = st.columns(4)
        for idx, (word, freq) in df_radar.iterrows():
            col = cols[idx % 4]
            col.metric(label=word, value=freq)

    # 6. 散点图（原生散点组件）
    elif chart_type == "散点图":
        st.subheader("TOP20词汇散点图")
        st.scatter_chart(df, x="词汇", y="词频", size="词频")

    # 7. 热力图（原生颜色块）
    elif chart_type == "热力图":
        st.subheader("TOP20词汇热力图")
        cols = st.columns(len(df))
        for idx, (word, freq) in enumerate(top20):
            col = cols[idx]
            col.write(word)
            col.markdown(f"<div style='background:#{int(255-freq*10):02x}e6f9; height:30px;'></div>", unsafe_allow_html=True)

    # 8. 漏斗图（原生分栏+高度渐变）
    elif chart_type == "漏斗图":
        st.subheader("TOP20词汇漏斗图")
        max_freq = df["词频"].max()
        for word, freq in top20:
            height = int(freq / max_freq * 50)
            st.markdown(f"<div style='background:#4285F4; height:{height}px; text-align:center; color:white; line-height:{height}px;'>{word} ({freq})</div>", unsafe_allow_html=True)

# 页面布局
st.title("📊 URL文本词频分析系统")
st.subheader("Streamlit原生版 | 无第三方依赖")

with st.sidebar:
    st.header("⚙️ 配置项")
    url = st.text_input("文章URL", value="https://www.guokr.com/article/440923/")
    min_freq = st.selectbox("最低词频过滤", [1,2,3,4,5])
    chart_type = st.selectbox("图表类型", ["词云图","柱状图","折线图","饼图","雷达图","散点图","热力图","漏斗图"])
    analyze_btn = st.button("🚀 抓取并分析")

# 分析逻辑
if analyze_btn:
    if not url:
        st.error("请输入有效的URL！")
    else:
        text = fetch_url_text(url)
        if text.startswith("URL抓取失败"):
            st.error(text)
        elif len(text) < 50:
            st.warning("URL文本过短，使用兜底测试文本！")
            text = BACKUP_TEXT
        
        word_freq, top20 = analyze_text(text, min_freq)
        if not top20:
            st.error("无有效词汇，降低词频重试！")
        else:
            st.success(f"分析成功！有效词汇{len(word_freq)}个")
            st.table([{"排名":i+1, "词汇":w, "词频":f} for i,(w,f) in enumerate(top20)])
            st.subheader(f"{chart_type}可视化")
            show_chart(top20, chart_type)
