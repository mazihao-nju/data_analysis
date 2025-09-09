import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "6"
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("cleaned_doctor_data.csv")

title_map = {'主任医师': 3, '副主任医师': 2, '主治医师': 1, '住院医师': 0}
df['title_level'] = df['title'].map(title_map).fillna(0)
df['log_price'] = np.log1p(df['price'])
df['log_comments'] = np.log1p(df['num_comments'])

features = [
    'reply_quality_score', 'service_attitude_score', 'reply_speed_score',
    'avg_star', 'log_price', 'log_comments', 'title_level'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.metrics import silhouette_score
inertia_list = []
silhouette_list = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=1000)
    labels = km.fit_predict(X_scaled)
    inertia_list.append(km.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_list.append(sil_score)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(K_range, inertia_list, 'o-', label='Inertia', color='blue')
ax2.plot(K_range, silhouette_list, 's--', label='Silhouette', color='green')
ax1.set_xlabel('聚类数 K')
ax1.set_ylabel('Inertia', color='blue')
ax2.set_ylabel('轮廓系数', color='green')
plt.title("K值选择：肘部法与轮廓系数对比")
fig.legend(loc='upper right')
plt.tight_layout()
plt.savefig("k_selection_metrics.png")
plt.close()

best_k = K_range[np.argmax(silhouette_list)]
print(f" 轮廓系数最佳聚类数 K = {best_k}，得分 = {max(silhouette_list):.4f}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=1000)
df['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette='Set2')
plt.title('医生聚类结果（PCA降维可视化）')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("pca_clusters.png")
plt.close()

grouped = df.groupby('cluster')[features].mean()
grouped_norm = (grouped - grouped.min()) / (grouped.max() - grouped.min())

labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 6))
for idx, row in grouped_norm.iterrows():
    values = row.tolist()
    values += values[:1]
    plt.plot(angles, values, label=f'簇 {idx}')
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], labels, fontsize=10)
plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2','0.4','0.6','0.8'])
plt.title("医生各聚类特征雷达图", fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("radar_plot.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(grouped.round(2), annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("不同聚类医生的特征均值热力图")
plt.xlabel("特征")
plt.ylabel("聚类簇编号")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()

df.to_csv("doctor_clusters.csv", index=False)


print("\n 每个簇的标签解释（自动生成）：")
desc_features = grouped.round(2)

for idx, row in desc_features.iterrows():
    desc = f"簇 {idx}: "
    desc += "评分高，" if row['avg_star'] > desc_features['avg_star'].mean() else "评分低，"
    desc += "回复快，" if row['reply_speed_score'] > desc_features['reply_speed_score'].mean() else ""
    desc += "服务态度好，" if row['service_attitude_score'] > desc_features['service_attitude_score'].mean() else ""
    desc += "问诊量大，" if row['log_comments'] > desc_features['log_comments'].mean() else "问诊量少，"
    desc += "价格高，" if row['log_price'] > desc_features['log_price'].mean() else "价格低，"
    desc += "职称高。" if row['title_level'] > desc_features['title_level'].mean() else "职称低。"
    print(desc)

print("\n 聚类分析全部完成！保存文件包括：")
print("doctor_clusters.csv, radar_plot.png, heatmap.png, pca_clusters.png, k_selection_metrics.png")

