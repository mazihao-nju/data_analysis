import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_comments_data.csv")
texts = df['comment_text'].dropna().astype(str).tolist()

stopwords = set([
    "的", "了", "是", "我", "也", "就", "在", "不", "都", "很", "和", "啊", "啦", "呢", "哦", "嘛", "呀",
    "吧", "这", "那", "你", "他", "她", "它", "我们", "他们", "一个", "没有", "看", "说", "还", "跟",
    "请问", "咨询", "医生", "感觉", "回复", "非常", "问题", "建议", "回答", "描述", "提出", "谢谢", "专业"
])

def clean_and_cut(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)  # 去除非中文字符
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w) > 1]

tokenized_texts = [clean_and_cut(text) for text in texts if len(text) > 4]
processed_texts = [' '.join(words) for words in tokenized_texts]

vectorizer = CountVectorizer(max_df=0.9, min_df=10)
X = vectorizer.fit_transform(processed_texts)
feature_names = vectorizer.get_feature_names_out()

k = 13
lda = LatentDirichletAllocation(n_components=k, random_state=42, learning_method='batch')
lda.fit(X)

print("\n 每个主题的关键词（Top 10）：\n")
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"主题 {topic_idx + 1}: {', '.join(top_words)}")
    topics.append(top_words)

topic_distributions = lda.transform(X)
df_topics = pd.DataFrame(topic_distributions, columns=[f"Topic_{i+1}" for i in range(k)])
df_result = df[['doctor_id', 'comment_text']].copy()
df_result = df_result.iloc[:df_topics.shape[0]]
df_result['main_topic'] = df_topics.idxmax(axis=1)
df_result.to_csv("comment_topics_k13.csv", index=False, encoding="utf-8-sig")
print("\n 主题建模完成，结果已保存：comment_topics_k13.csv")

