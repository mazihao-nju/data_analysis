import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "6"

from scipy.stats import chi2_contingency

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("doctor_clusters.csv")

region_map = {
    '山东': '山河四省', '山西': '山河四省', '河南': '山河四省', '河北': '山河四省',
    '江苏': '江浙沪', '浙江': '江浙沪', '上海': '江浙沪',
    '四川': '川渝', '重庆': '川渝'
}
df['region_group'] = df['city'].map(region_map)
df = df.dropna(subset=['region_group'])

region_cluster_ct = pd.crosstab(df['region_group'], df['cluster'])
region_cluster_ratio = region_cluster_ct.div(region_cluster_ct.sum(axis=1), axis=0)

plt.figure(figsize=(8, 5))
region_cluster_ct.plot(kind='bar', stacked=True, colormap='Set2')
plt.title("各区域医生在不同聚类中的分布")
plt.xlabel("区域")
plt.ylabel("医生数量")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("region_group_stacked_bar.png")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(region_cluster_ratio, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title("区域-聚类分布比例热力图")
plt.xlabel("聚类簇编号")
plt.ylabel("区域")
plt.tight_layout()
plt.savefig("region_group_heatmap.png")
plt.close()

chi2, p, dof, expected = chi2_contingency(region_cluster_ct)
print("\n 区域-聚类卡方检验结果：")
print(f"Chi2 = {chi2:.2f}, 自由度 = {dof}, p 值 = {p:.4f}")
if p < 0.05:
    print(" 结论：区域分布与医生聚类之间具有统计学显著关联 (p < 0.05)")
else:
    print(" 结论：区域分布与医生聚类之间无显著关联 (p >= 0.05)")

feature_cols = [
    'reply_quality_score', 'service_attitude_score', 'reply_speed_score',
    'avg_star', 'price', 'num_comments', 'title_level'
]
region_feature_mean = df.groupby('region_group')[feature_cols].mean().round(2)

plt.figure(figsize=(10, 6))
sns.heatmap(region_feature_mean, annot=True, fmt=".2f", cmap="RdPu")
plt.title("不同区域医生特征均值热力图")
plt.tight_layout()
plt.savefig("region_group_feature_heatmap.png")
plt.close()

print("\n 已保存区域堆叠图、热力图和特征均值分析结果。")
