# -*- coding: utf-8 -*-
from flask import Flask, render_template_string, request, flash
import jieba
import re
from collections import Counter
import pyecharts.options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, Funnel
from pyecharts.globals import ThemeType
import requests
from bs4 import BeautifulSoup
import webbrowser
import threading

app = Flask(__name__)
app.secret_key = "text_analysis_32bit"
analysis_data = {"top20": [], "word_freq": {}, "min_freq": 1, "chart_type": "词云图"}
BACKUP_TEXT = """人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学。它涵盖了机器学习、自然语言处理、计算机视觉、专家系统等多个领域。机器学习是人工智能的核心，通过让计算机从数据中学习模式，而无需显式编程。深度学习作为机器学习的一个分支，使用神经网络模拟人脑结构，在图像识别、语音识别等领域取得了突破性进展。自然语言处理则专注于让计算机理解和生成人类语言，如聊天机器人、机器翻译等应用。人工智能的发展已经深刻影响了医疗、金融、交通、教育等各行各业，未来还将继续推动社会的数字化转型。"""

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

def analyze_text(text, min_freq=1):
    stop_words = {'的','了','是','在','和','有','就','都','这','那','个','为','把','我','你','他','她','它','我们','你们','他们','这里','那里','什么','怎么','为什么','如何','然后','但是','如果','因为','所以','虽然','既然','之','于','也','还','及','与','或','即','所','将','会','可','能','应','该','要','需','须','得','过','着','啊','呀','呢','吗','吧'}
    words = [w for w in jieba.lcut(re.sub(r'\s+', ' ', text)) if w not in stop_words and len(w)>1]
    word_freq = Counter(words)
    return {k:v for k,v in word_freq.items() if v>=min_freq}, sorted(word_freq.items(), key=lambda x:x[1], reverse=True)[:20]

def generate_chart(top20, chart_type):
    if not top20: return "<div style='text-align:center;padding:50px;color:#666;'>暂无有效数据</div>"
    words, freqs = [i[0] for i in top20], [i[1] for i in top20]
    max_freq = max(freqs) if freqs else 1
    
    if chart_type == "词云图":
        c = WordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add("词频", top20, word_size_range=[20,80]).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇词云图"))
    elif chart_type == "柱状图":
        c = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add_xaxis(words).add_yaxis("词频", freqs).reversal_axis().set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇柱状图"))
    elif chart_type == "折线图":
        c = Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add_xaxis(words).add_yaxis("词频", freqs).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇折线图"))
    elif chart_type == "饼图":
        c = Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add("", list(zip(words,freqs)), radius=["30%","70%"]).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇饼图"))
    elif chart_type == "雷达图":
        c = Radar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add_schema(schema=[{"name":w,"max":max_freq} for w in words[:8]]).add("词频", [freqs[:8]]).set_global_opts(title_opts=opts.TitleOpts(title="TOP8词汇雷达图"))
    elif chart_type == "散点图":
        c = Scatter(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add_xaxis(words).add_yaxis("词频", freqs, symbol_size=lambda x:x*5).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇散点图"), visualmap_opts=opts.VisualMapOpts(max_=max_freq))
    elif chart_type == "热力图":
        c = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add_xaxis(words).add_yaxis("词频", ["频次"], [[i,0,v] for i,v in enumerate(freqs)]).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇热力图"), visualmap_opts=opts.VisualMapOpts(max_=max_freq))
    elif chart_type == "漏斗图":
        c = Funnel(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="500px")).add("词频", top20).set_global_opts(title_opts=opts.TitleOpts(title="TOP20词汇漏斗图"))
    return c.render_embed()

HTML_TPL = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URL词频分析系统</title>
    <style>
        * {margin:0;padding:0;box-sizing:border-box;font-family:"Microsoft YaHei",sans-serif;}
        body {background:#f5f7fa;color:#333;line-height:1.6;}
        .container {max-width:1200px;margin:0 auto;padding:20px;}
        .header {text-align:center;margin-bottom:30px;padding-bottom:20px;border-bottom:1px solid #e0e0e0;}
        .header h1 {color:#2d3748;font-size:32px;margin-bottom:10px;}
        .header p {color:#718096;font-size:16px;}
        .form-container {background:white;padding:25px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:20px;}
        .form-group {display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:15px;align-items:end;margin-bottom:0;}
        @media (max-width:768px) {.form-group {grid-template-columns:1fr;}}
        label {display:block;margin-bottom:8px;font-weight:600;color:#2d3748;font-size:14px;}
        input, select {width:100%;padding:12px 15px;border:1px solid #e0e0e0;border-radius:8px;font-size:14px;}
        input:focus, select:focus {outline:none;border-color:#4285F4;box-shadow:0 0 0 3px rgba(66,133,244,0.1);}
        button {width:100%;padding:12px 15px;background:#4285F4;color:white;border:none;border-radius:8px;font-weight:600;font-size:15px;cursor:pointer;}
        button:hover {background:#3367d6;}
        .main {display:flex;gap:20px;flex-wrap:wrap;}
        .sidebar {flex:1;min-width:300px;background:white;padding:25px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.05);}
        .content {flex:2;min-width:600px;background:white;padding:25px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.05);}
        .alert {padding:15px;margin-bottom:20px;border-radius:8px;font-size:14px;}
        .alert-success {background:#e8f5e9;color:#2e7d32;border:1px solid #c8e6c9;}
        .alert-error {background:#ffebee;color:#c62828;border:1px solid #ffcdd2;}
        .alert-warning {background:#fff8e1;color:#ff8f00;border:1px solid #ffecb3;}
        .metric-cards {display:flex;gap:15px;margin-bottom:25px;flex-wrap:wrap;}
        .metric-card {flex:1;min-width:100px;padding:15px;background:#e8f4f8;border-radius:8px;text-align:center;}
        .metric-card h3 {color:#2d3748;font-size:24px;margin-bottom:5px;}
        .metric-card p {color:#718096;font-size:14px;}
        .result-table {width:100%;border-collapse:collapse;margin-top:20px;}
        .result-table th, .result-table td {padding:12px;text-align:left;border-bottom:1px solid #f0f0f0;}
        .result-table th {background:#f8f9fa;color:#2d3748;font-weight:600;}
    </style>
    <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 URL文本词频分析系统</h1>
            <p>仅URL抓取 | 8种可视化图表 | 32位Windows兼容</p>
        </div>
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}{% for c,m in messages %}<div class="alert alert-{{c}}">{{m}}</div>{% endfor %}{% endif %}
        {% endwith %}
        <div class="form-container">
            <form method="POST">
                <div class="form-group">
                    <div><label for="url">文章URL</label><input type="text" id="url" name="url" placeholder="输入中文文章URL" value="https://www.guokr.com/article/440923/" required></div>
                    <div><label for="min_freq">最低词频过滤</label><select id="min_freq" name="min_freq">{% for i in range(1,6) %}<option value="{{i}}" {% if i==analysis_data.min_freq %}selected{% endif %}>{{i}}</option>{% endfor %}</select></div>
                    <div><label for="chart_type">图表类型</label><select id="chart_type" name="chart_type">{% for t in ["词云图","柱状图","折线图","饼图","雷达图","散点图","热力图","漏斗图"] %}<option value="{{t}}" {% if t==analysis_data.chart_type %}selected{% endif %}>{{t}}</option>{% endfor %}</select></div>
                    <div><button type="submit">抓取并分析</button></div>
                </div>
            </form>
        </div>
        <div class="main">
            <div class="sidebar">
                {% if analysis_data.top20 %}
                    <div class="metric-cards">
                        <div class="metric-card"><h3>{{analysis_data.word_freq|length}}</h3><p>有效词汇总数</p></div>
                        <div class="metric-card"><h3>{{analysis_data.top20[0][1]}}</h3><p>最高词频</p></div>
                        <div class="metric-card"><h3>20</h3><p>展示词汇数</p></div>
                    </div>
                    <h3 style="margin-bottom:15px;color:#2d3748;">TOP20词汇</h3>
                    <table class="result-table">
                        <thead><tr><th>排名</th><th>词汇</th><th>词频</th></tr></thead>
                        <tbody>{% for idx, (w,f) in enumerate(analysis_data.top20,1) %}<tr><td>{{idx}}</td><td>{{w}}</td><td>{{f}}</td></tr>{% endfor %}</tbody>
                    </table>
                {% else %}
                    <div style="text-align:center;padding:50px;color:#666;">输入URL并点击分析按钮开始使用</div>
                {% endif %}
            </div>
            <div class="content">
                <h2 style="margin-bottom:20px;color:#2d3748;">{{analysis_data.chart_type}}可视化</h2>
                {{chart_html|safe}}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    global analysis_data
    if request.method == 'POST':
        url = request.form.get('url').strip()
        min_freq = int(request.form.get('min_freq',1))
        chart_type = request.form.get('chart_type','词云图')
        
        if not url:
            flash("请输入有效的URL！", "error")
        else:
            text = fetch_url_text(url)
            if text.startswith("URL抓取失败"):
                flash(text, "error")
            elif len(text) < 50:
                flash(f"URL文本过短（{len(text)}字），使用兜底文本！", "warning")
                text = BACKUP_TEXT
            
            word_freq, top20 = analyze_text(text, min_freq)
            if not top20:
                flash("无有效词汇，降低词频重试！", "error")
            else:
                analysis_data = {"top20":top20, "word_freq":word_freq, "min_freq":min_freq, "chart_type":chart_type}
                flash(f"分析成功！有效词汇{len(word_freq)}个，展示：{chart_type}", "success")
        
        chart_html = generate_chart(analysis_data["top20"], analysis_data["chart_type"])
        return render_template_string(HTML_TPL, analysis_data=analysis_data, chart_html=chart_html, enumerate=enumerate)
    
    chart_html = generate_chart(analysis_data["top20"], analysis_data["chart_type"])
    return render_template_string(HTML_TPL, analysis_data=analysis_data, chart_html=chart_html, enumerate=enumerate)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    jieba.initialize()
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False, threaded=True)
