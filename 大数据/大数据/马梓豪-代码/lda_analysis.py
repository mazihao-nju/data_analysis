import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("comment_topics_k13.csv")


topic_counts = df['main_topic'].value_counts(normalize=True).sort_index()
topic_counts.plot(kind='bar', figsize=(10, 5))
plt.title("每个主题在总体评论中的占比")
plt.ylabel("占比")
plt.xlabel("主题编号")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("topic_distribution_overall.png")
plt.show()

print("\n 每个主题占比：")
print(topic_counts)

def classify_sentiment(text):
    positive_keywords = ["感谢", "感激", "耐心", "态度好", "热情", "认真", "效果好", "好转", "回复及时", "医术高明"]
    for word in positive_keywords:
        if word in str(text):
            return "正面"
    return "一般"

df['sentiment'] = df['comment_text'].apply(classify_sentiment)

positive_topics = df[df['sentiment'] == "正面"]['main_topic'].value_counts(normalize=True).sort_index()
neutral_topics = df[df['sentiment'] == "一般"]['main_topic'].value_counts(normalize=True).sort_index()

compare_df = pd.DataFrame({
    "正面评论": positive_topics,
    "一般评论": neutral_topics
}).fillna(0)

compare_df.plot(kind='bar', figsize=(12, 6))
plt.title("正面评论 vs 一般评论的主题分布对比")
plt.xlabel("主题编号")
plt.ylabel("主题占比")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("topic_distribution_comparison.png")
plt.show()

compare_df['差异'] = compare_df['正面评论'] - compare_df['一般评论']
print("\n 差异最大的前5个主题：")
print(compare_df['差异'].sort_values(ascending=False).head(5))
print("\n 正面评论占比较低的主题（可能为不满点）：")
print(compare_df['差异'].sort_values().head(5))
