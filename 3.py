# -*- coding: utf-8 -*-
import streamlit as st
import jieba
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.font_manager as fm
import os

# 页面配置 + 加载中文字体（核心：解决中文方块问题）
st.set_page_config(page_title="URL词频分析系统", page_icon="📊", layout="wide")
# 加载仓库中的SimHei.ttf字体
font_path = os.path.join(os.path.dirname(__file__), 'SimHei.ttf')
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100

# 兜底文本
BACKUP_TEXT = """人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学。它涵盖了机器学习、自然语言处理、计算机视觉、专家系统等多个领域。机器学习是人工智能的核心，通过让计算机从数据中学习模式，而无需显式编程。深度学习作为机器学习的一个分支，使用神经网络模拟人脑结构，在图像识别、语音识别等领域取得了突破性进展。自然语言处理则专注于让计算机理解和生成人类语言，如聊天机器人、机器翻译等应用。人工智能的发展已经深刻影响了医疗、金融、交通、教育等各行各业，未来还将继续推动社会的数字化转型。"""

# 1. URL文本抓取（不变）
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

# 2. 文本分析（不变）
def analyze_text(text, min_freq=1):
    stop_words = {'的','了','是','在','和','有','就','都','这','那','个','为','把','我','你','他','她','它','我们','你们','他们','这里','那里','什么','怎么','为什么','如何','然后','但是','如果','因为','所以','虽然','既然','之','于','也','还','及','与','或','即','所','将','会','可','能','应','该','要','需','须','得','过','着','啊','呀','呢','吗','吧'}
    words = [w for w in jieba.lcut(re.sub(r'\s+', ' ', text)) if w not in stop_words and len(w)>1]
    word_freq = Counter(words)
    return {k:v for k,v in word_freq.items() if v>=min_freq}, sorted(word_freq.items(), key=lambda x:x[1], reverse=True)[:20]

# 3. 雷达图投影配置（不变）
def radar_polar(theta):
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)
        def _gen_axes_spines(self):
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}
    def unit_poly_verts(theta):
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts
    register_projection(RadarAxes)
    return theta

# 4. 逐图完善的图表生成函数（每个图表都用fontproperties=font_prop）
def show_chart(top20, chart_type):
    if not top20:
        st.warning("暂无有效数据可展示")
        return
    
    df = pd.DataFrame(top20, columns=["词汇", "词频"])
    colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#9C27B0', '#00ACC1', '#FF7043', '#607D8B']

    # 1. 词云图
    if chart_type == "词云图":
        fig, ax = plt.subplots(figsize=(12, 8))
        df_sorted = df.sort_values('词频', ascending=True)
        sizes = df_sorted['词频'] * 10
        for i, (word, freq, size) in enumerate(zip(df_sorted['词汇'], df_sorted['词频'], sizes)):
            ax.text(
                np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9),
                word, fontsize=size/2, color=np.random.choice(colors),
                ha='center', va='center', rotation=np.random.uniform(-30, 30),
                fontproperties=font_prop  # 中文显示
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("TOP20词汇词云图", fontsize=16, pad=20, fontproperties=font_prop)
        st.pyplot(fig)
    
    # 2. 柱状图
    elif chart_type == "柱状图":
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df['词频'], color=colors[0], alpha=0.8)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=10, fontproperties=font_prop)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['词汇'], fontsize=11, fontproperties=font_prop)
        ax.set_xlabel("词频", fontsize=12, fontproperties=font_prop)
        ax.set_ylabel("词汇", fontsize=12, fontproperties=font_prop)
        ax.set_title("TOP20词汇柱状图", fontsize=16, pad=20, fontproperties=font_prop)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    # 3. 折线图
    elif chart_type == "折线图":
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df['词汇'], df['词频'], marker='o', linewidth=2.5, color=colors[0], 
                markersize=8, markerfacecolor=colors[1], markeredgecolor='white', markeredgewidth=2)
        ax.fill_between(df['词汇'], df['词频'], alpha=0.2, color=colors[0])
        for x, y in zip(df['词汇'], df['词频']):
            ax.text(x, y + 0.2, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontproperties=font_prop)
        ax.set_xlabel("词汇", fontsize=12, fontproperties=font_prop)
        ax.set_ylabel("词频", fontsize=12, fontproperties=font_prop)
        ax.set_title("TOP20词汇折线图", fontsize=16, pad=20, fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    # 4. 饼图
    elif chart_type == "饼图":
        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, texts, autotexts = ax.pie(
            df['词频'], labels=df['词汇'], autopct='%1.1f%%',
            colors=colors*3, startangle=90, textprops={'fontsize': 10, 'fontproperties': font_prop}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title("TOP20词汇饼图", fontsize=16, pad=20, fontproperties=font_prop)
        ax.legend(wedges, df['词汇'], title="词汇", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop=font_prop)
        st.pyplot(fig)
    
    # 5. 雷达图
    elif chart_type == "雷达图":
        df_radar = df.head(8)
        N = len(df_radar)
        theta = radar_polar(np.linspace(0, 2*np.pi, N, endpoint=False))
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
        values = df_radar['词频'].values
        ax.plot(theta, values, color=colors[0], linewidth=2, label='词频')
        ax.fill(theta, values, color=colors[0], alpha=0.2)
        ax.set_varlabels(df_radar['词汇'])
        ax.set_ylim(0, df['词频'].max() + 1)
        ax.set_title("TOP8词汇雷达图", fontsize=16, pad=20, fontproperties=font_prop)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # 6. 散点图
    elif chart_type == "散点图":
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df['词汇'], df['词频'], 
            s=df['词频']*50, c=df['词频'], cmap='Blues', 
            alpha=0.7, edgecolors='white', linewidth=1
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('词频', fontsize=11, fontproperties=font_prop)
        for x, y in zip(df['词汇'], df['词频']):
            ax.text(x, y + 0.2, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontproperties=font_prop)
        ax.set_xlabel("词汇", fontsize=12, fontproperties=font_prop)
        ax.set_ylabel("词频", fontsize=12, fontproperties=font_prop)
        ax.set_title("TOP20词汇散点图", fontsize=16, pad=20, fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    # 7. 热力图
    elif chart_type == "热力图":
        fig, ax = plt.subplots(figsize=(14, 4))
        heat_data = df['词频'].values.reshape(1, -1)
        im = ax.imshow(heat_data, cmap='Blues', aspect='auto')
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df['词汇'], fontsize=10, fontproperties=font_prop)
        ax.set_yticks([0])
        ax.set_yticklabels(["词频"], fontsize=11, fontproperties=font_prop)
        for i in range(len(df)):
            text = ax.text(i, 0, f'{int(heat_data[0][i])}',
                           ha="center", va="center", color="black", fontsize=9, fontproperties=font_prop)
        ax.set_title("TOP20词汇热力图", fontsize=16, pad=20, fontproperties=font_prop)
        plt.colorbar(im, ax=ax, label='词频', fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        st.pyplot(fig)
    
    # 8. 漏斗图
    elif chart_type == "漏斗图":
        fig, ax = plt.subplots(figsize=(12, 8))
        df_funnel = df.sort_values('词频', ascending=False)
        max_width = 0.8
        widths = df_funnel['词频'] / df_funnel['词频'].max() * max_width
        y_pos = np.arange(len(df_funnel))
        for i, (word, freq, width) in enumerate(zip(df_funnel['词汇'], df_funnel['词频'], widths)):
            rect = plt.Rectangle((0.5 - width/2, i), width, 0.8, 
                                facecolor=colors[i%len(colors)], alpha=0.7, edgecolor='white')
            ax.add_patch(rect)
            ax.text(0.5, i + 0.4, f'{word} ({freq})', ha='center', va='center', 
                    fontsize=10, fontweight='bold', fontproperties=font_prop)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(df_funnel))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title("TOP20词汇漏斗图", fontsize=16, pad=20, fontproperties=font_prop)
        st.pyplot(fig)

# ======== Streamlit页面布局（所有中文都用font_prop） ========
st.title("📊 URL文本词频分析系统", font=font_prop)
st.subheader("Streamlit Cloud部署版 | 8种图表精准显示", font=font_prop)

# 输入区域
with st.sidebar:
    st.header("⚙️ 配置项", font=font_prop)
    url = st.text_input("文章URL", value="https://www.guokr.com/article/440923/", placeholder="输入公开中文文章URL")
    min_freq = st.selectbox("最低词频过滤", options=[1,2,3,4,5], index=0)
    chart_type = st.selectbox(
        "图表类型", 
        options=["词云图","柱状图","折线图","饼图","雷达图","散点图","热力图","漏斗图"], 
        index=0
    )
    analyze_btn = st.button("🚀 抓取并分析", type="primary")

# 分析逻辑（不变）
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
        
        word_freq, top20 = analyze_text(text, min_freq)
        if not top20:
            st.error("❌ 无有效词汇，降低词频重试！")
        else:
            st.success(f"✅ 分析成功！有效词汇{len(word_freq)}个，展示：{chart_type}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("有效词汇总数", len(word_freq))
            with col2:
                st.metric("最高词频", top20[0][1])
            with col3:
                st.metric("展示词汇数", 20)
            
            st.subheader("📋 TOP20词汇列表", font=font_prop)
            st.table([{"排名":i+1, "词汇":w, "词频":f} for i,(w,f) in enumerate(top20)])
            
            st.subheader(f"📈 {chart_type}可视化", font=font_prop)
            show_chart(top20, chart_type)

# 页脚
st.divider()
st.caption("💡 部署于Streamlit Cloud | 8种图表100%精准显示", font=font_prop)
