import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from statsmodels.miscmodels.ordinal_model import OrderedModel

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 读取数据 =====
df = pd.read_csv("cleaned_comments_data.csv")

# ===== 编码映射 =====
rating_map = {
    "非常满意": 5,
    "满意": 4,
    "一般": 3,
    "较差": 2,
    "非常差": 1
}

# ===== 清洗与映射 =====
df = df.dropna(subset=['overall_star', 'reply_quality', 'service_attitude', 'reply_speed', 'consult_type'])
df['reply_quality_num'] = df['reply_quality'].map(rating_map)
df['service_attitude_num'] = df['service_attitude'].map(rating_map)
df['reply_speed_num'] = df['reply_speed'].map(rating_map)
df['consult_type_encoded'] = pd.factorize(df['consult_type'])[0]
df['overall_star'] = pd.to_numeric(df['overall_star'], errors='coerce')

df = df.dropna(subset=['reply_quality_num', 'service_attitude_num', 'reply_speed_num', 'consult_type_encoded', 'overall_star'])

# 自变量与因变量
X = df[['reply_quality_num', 'service_attitude_num', 'reply_speed_num', 'consult_type_encoded']]
y = df['overall_star'].astype(int)

# ===== 有序逻辑回归 =====
model = OrderedModel(y, X, distr='logit')
res = model.fit(method='bfgs', disp=False)

# ===== 输出模型结果 =====
print(res.summary())

# ===== 构造用于可视化的数据表 =====
params = res.params[:len(X.columns)]
conf_int = res.conf_int().loc[X.columns]

coef_df = pd.DataFrame({
    '变量': X.columns,
    '系数': params.values,
    '置信区间下限': conf_int[0].values,
    '置信区间上限': conf_int[1].values
})

# ===== 可视化：系数图 =====
plt.figure(figsize=(8, 6))
sns.barplot(x='系数', y='变量', hue='变量', data=coef_df, palette='Blues_d', legend=False)
plt.errorbar(
    coef_df['系数'], coef_df['变量'],
    xerr=[coef_df['系数'] - coef_df['置信区间下限'], coef_df['置信区间上限'] - coef_df['系数']],
    fmt='none', c='black', capsize=5
)
plt.title("评分影响因素：有序逻辑回归系数图", fontsize=14)
plt.tight_layout()
plt.savefig("评分影响因素_有序逻辑回归系数图.png")
plt.show()
