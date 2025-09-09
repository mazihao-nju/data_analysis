import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# === Step 1: 读取数据 ===
df = pd.read_csv("cleaned_comments_data.csv")

# === Step 2: 文本字段数值映射 ===
satisfaction_map = {
    "非常满意": 5,
    "满意": 4,
    "一般": 3,
    "较差": 2,
    "非常差": 1
}

# 去除关键字段缺失
df = df.dropna(subset=['overall_star', 'consult_type', 'reply_quality', 'service_attitude', 'reply_speed'])

# 映射文本变量为数值
df['reply_quality_num'] = df['reply_quality'].map(satisfaction_map)
df['service_attitude_num'] = df['service_attitude'].map(satisfaction_map)
df['reply_speed_num'] = df['reply_speed'].map(satisfaction_map)
df['consult_type_code'] = pd.factorize(df['consult_type'])[0]  # 编码：图文0，视频1，健康2等

# 再次清除无效映射
df = df.dropna(subset=['reply_quality_num', 'service_attitude_num', 'reply_speed_num'])

# === Step 3: 构造模型输入 ===
X = df[['reply_quality_num', 'service_attitude_num', 'reply_speed_num', 'consult_type_code']]
X = sm.add_constant(X)
y = df['overall_star'].astype(int)  # 评分为1~5

# === Step 4: 拟合多元逻辑回归（Softmax） ===
model = sm.MNLogit(y, X).fit()
print(model.summary())

# === Step 5: 可视化系数 ===
params = model.params.T  # 行为变量，列为类别（评分1~5中的任意4个）
params = params.drop("const", errors="ignore")
params.index.name = "变量"

# 手动设置颜色，确保五类都能显示（如果模型中评分缺少某一类，颜色可能仍会少）
rating_classes = params.columns.tolist()
palette = sns.color_palette("Set2", len(rating_classes))

# === Step 6: 画图 ===
plt.figure(figsize=(10, 6))
for i, col in enumerate(rating_classes):
    sns.barplot(x=params[col], y=params.index, label=f"评分={col}", orient='h', color=palette[i])

plt.title("各变量对不同评分的回归系数（Softmax）", fontsize=14)
plt.xlabel("回归系数")
plt.ylabel("变量")
plt.legend(title="评分类别")
plt.tight_layout()
plt.savefig("多元逻辑回归_评分影响因素.png")
plt.show()
