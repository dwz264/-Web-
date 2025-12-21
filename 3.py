# -*- coding: utf-8 -*-
# streamlit_text_analysis.py (核心文件)
import streamlit as st
import jieba
import re
from collections import Counter
import pyecharts.options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, Funnel
from pyecharts.globals import ThemeType
import requests
from bs4 import BeautifulSoup
from streamlit.components.v1 import html

# 页面配置
st.set_page_config(
    page_title="URL文本词频分析系统",
    page_icon="📊",
    layout="wide"
)

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

# 3. 生成图表
def generate_chart_html(top20, chart_type):
    if not top20:
        return "<div style='text-align:center;padding:50px;color:#666;'>暂无有效数据</div>"
    
    words, freqs = [i[0] for i in top20], [i[1] for i in top20]
    max_freq = max(freqs) if freqs else 1

    # 生成图表（改用render_notebook，适配Streamlit）
    if chart_type == "词云图":
        c = WordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add("词频", top20, word_size_range=[20, 80])
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇词云图"))
    elif chart_type == "柱状图":
        c = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add_xaxis(words)
        c.add_yaxis("词频", freqs)
        c.reversal_axis()
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇柱状图"))
    elif chart_type == "折线图":
        c = Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add_xaxis(words)
        c.add_yaxis("词频", freqs)
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇折线图"))
    elif chart_type == "饼图":
        c = Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add("", list(zip(words, freqs)), radius=["30%", "70%"])
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇饼图"))
    elif chart_type == "雷达图":
        c = Radar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add_schema(schema=[{"name": w, "max": max_freq} for w in words[:8]])
        c.add("词频", [freqs[:8]])
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP8词汇雷达图"))
    elif chart_type == "散点图":
        c = Scatter(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add_xaxis(words)
        c.add_yaxis("词频", freqs, symbol_size=lambda x: x*5)
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇散点图"), visualmap_opts=opts.VisualMapOpts(max_=max_freq))
    elif chart_type == "热力图":
        c = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add_xaxis(words)
        c.add_yaxis("词频", ["频次"], [[i, 0, v] for i, v in enumerate(freqs)])
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇热力图"), visualmap_opts=opts.VisualMapOpts(max_=max_freq))
    elif chart_type == "漏斗图":
        c = Funnel(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800", height="500"))
        c.add("词频", top20)
        c.set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇漏斗图"))

    # 生成适合Streamlit的HTML（关键修复）
    return c.render_notebook()
# ======== Streamlit页面布局 ========
st.title("📊 URL文本词频分析系统")
st.subheader("Streamlit Cloud部署版 | 支持8种图表可视化")

# 输入区域
with st.sidebar:
    st.header("⚙️ 配置项")
    url = st.text_input("文章URL", value="https://www.guokr.com/article/440923/", placeholder="输入公开中文文章URL")
    min_freq = st.selectbox("最低词频过滤", options=[1,2,3,4,5], index=0)
    chart_type = st.selectbox("图表类型", options=["词云图","柱状图","折线图","饼图","雷达图","散点图","热力图","漏斗图"], index=0)
    analyze_btn = st.button("🚀 抓取并分析", type="primary")

# 分析逻辑
if analyze_btn:
    if not url:
        st.error("❌ 请输入有效的URL！")
    else:
        with st.spinner("🔍 正在抓取URL文本..."):
            text = fetch_url_text(url)
        
        if text.startswith("URL抓取失败"):
            st.error(f"❌ {text}")
        elif len(text) < 50:
            st.warning(f"⚠️ URL文本过短（{len(text)}字），使用兜底测试文本！")
            text = BACKUP_TEXT
        
        # 分词分析
        word_freq, top20 = analyze_text(text, min_freq)
        if not top20:
            st.error("❌ 无有效词汇，降低词频重试！")
        else:
            st.success(f"✅ 分析成功！有效词汇{len(word_freq)}个，展示：{chart_type}")
            
            # 展示统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("有效词汇总数", len(word_freq))
            with col2:
                st.metric("最高词频", top20[0][1])
            with col3:
                st.metric("展示词汇数", 20)
            
            # 展示TOP20表格
            st.subheader("📋 TOP20词汇列表")
            st.table([{"排名":i+1, "词汇":w, "词频":f} for i,(w,f) in enumerate(top20)])
            
            # 展示图表
            st.subheader(f"📈 {chart_type}可视化")
            chart_html = generate_chart_html(top20, chart_type)
            html(chart_html, width=850, height=550)

# 页脚
st.divider()

st.caption("💡 部署于Streamlit Cloud | 支持32位系统兼容")

