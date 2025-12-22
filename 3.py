import re
import jieba
import os
from collections import Counter


# 1. 文本清洗函数（过滤HTML标签、标点、冗余文本）
def clean_text(raw_text):
    # 移除HTML标签
    html_pattern = re.compile(r'<[^>]+>', re.S)
    text_no_html = html_pattern.sub('', raw_text)
    
    # 移除“网页标题”“网页链接”等冗余固定文本
    redundant_pattern = re.compile(r'网页标题|网页链接', re.S)
    text_no_redundant = redundant_pattern.sub('', text_no_html)
    
    # 仅保留中文汉字，过滤标点/数字/特殊符号
    punctuation_pattern = re.compile(r'[^\u4e00-\u9fa5]', re.S)
    text_no_punct = punctuation_pattern.sub(' ', text_no_redundant)
    
    # 移除多余空格
    text_clean = re.sub(r'\s+', ' ', text_no_punct).strip()
    return text_clean

# 2. 读取文件函数
def read_file(filename):
    if not os.path.exists(filename):
        print(f"错误：未找到文件 {filename}！")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ 成功读取 {filename}")
        return content
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        return None

# 3. 分词+词频统计函数
def word_analysis(clean_text):
    # 扩充停用词表（过滤导航/无意义词汇）
    stop_words = {
        '的', '了', '是', '在', '和', '有', '就', '都', '这', '那', '个', '为', '把',
        '首页', '链接', '网页', '登录', '退出', '报名', '申请', '通知公告', '竞赛通知',
        '新闻首页', '大赛新闻', '学会新闻', '精彩回顾', '竞赛作品', '校赛申请', '关于', '年', '届'
    }
    
    # jieba精确分词
    word_list = jieba.lcut(clean_text)
    # 过滤停用词+单字
    filtered_words = [w for w in word_list if w not in stop_words and len(w) > 1]
    
    # 统计词频，取TOP20
    word_freq = Counter(filtered_words)
    top20_words = word_freq.most_common(20)
    
    return filtered_words, top20_words  # 返回：分词结果、TOP20词频

# 4. 保存结果（分词结果换行显示，仅保留TOP20+分词结果）
def save_results(filtered_words, top20_words):
    with open('words.txt', 'w', encoding='utf-8') as f:
        # 第一部分：完整分词结果（每行显示10个词，换行）
        f.write("===== 文本分词结果 =====\n")
        # 每10个分词换一行，提升可读性
        line_words = []
        for idx, word in enumerate(filtered_words, 1):
            line_words.append(word)
            # 每10个词换行，或最后不足10个词时换行
            if idx % 10 == 0 or idx == len(filtered_words):
                f.write(' '.join(line_words) + "\n")
                line_words = []
        
        # 空行分隔，提升格式整洁度
        f.write("\n")
        
        # 第二部分：TOP20高频词
        f.write("===== 词频最高的20个词 =====\n")
        for idx, (word, count) in enumerate(top20_words, 1):
            f.write(f"{idx:2d}. {word:<8} 出现次数：{count}\n")
    
    print("\n✅ words.txt已生成（分词结果换行显示）")
    # 控制台输出TOP20
    print("\n===== TOP20 高频词 =====")
    for idx, (word, count) in enumerate(top20_words, 1):
        print(f"{idx:2d}. {word:<8} 出现次数：{count}")

# 主函数
def main():
    target_file = "new1.txt"
    # 读取文件
    raw_content = read_file(target_file)
    if not raw_content:
        return
    
    # 清洗文本
    clean_content = clean_text(raw_content)
    
    # 分词+统计TOP20
    filtered_words, top20_words = word_analysis(clean_content)
    
    # 保存结果（分词换行+TOP20）
    save_results(filtered_words, top20_words)
    print("\n🎉 处理完成！")

if __name__ == '__main__':
    jieba.initialize()
    main()
