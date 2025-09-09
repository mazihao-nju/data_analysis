import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def clean_and_cut(text, stopwords):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w) > 1]

def main():

    df = pd.read_csv("cleaned_comments_data.csv")
    texts = df['comment_text'].dropna().astype(str).tolist()


    stopwords = set([
        "的", "了", "是", "我", "也", "就", "在", "不", "都", "很", "和", "啊", "啦", "呢", "哦", "嘛", "呀",
        "吧", "这", "那", "你", "他", "她", "它", "我们", "他们", "一个", "没有", "看", "说", "还", "跟",
        "请问", "咨询", "医生", "感觉", "回复", "非常", "问题", "建议", "回答", "描述", "提出", "谢谢", "专业"
    ])

    tokenized_texts = [clean_and_cut(text, stopwords) for text in texts if len(text) > 4]
    processed_texts = [' '.join(words) for words in tokenized_texts]


    vectorizer = CountVectorizer(max_df=0.9, min_df=10)
    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()


    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    coherence_scores = []
    perplexity_scores = []
    topic_range = range(2, 16)

    for k in topic_range:
        print(f"计算中：主题数 = {k}")
        lda_gensim = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=10,
            random_state=42
        )
        cm = CoherenceModel(model=lda_gensim, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())

        lda_sklearn = LatentDirichletAllocation(n_components=k, random_state=42, learning_method='batch')
        lda_sklearn.fit(X)
        perplexity_scores.append(lda_sklearn.perplexity(X))


    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('主题数')
    ax1.set_ylabel('一致性得分 (c_v)', color=color)
    ax1.plot(topic_range, coherence_scores, marker='o', color=color, label='一致性')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('困惑度 (perplexity)', color=color)
    ax2.plot(topic_range, perplexity_scores, marker='s', linestyle='--', color=color, label='困惑度')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("不同主题数下的一致性与困惑度")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lda_k_coherence_perplexity.png")
    plt.show()

if __name__ == "__main__":
    main()
