import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


df = pd.read_csv("cleaned_doctor_data.csv")
df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
df = df.dropna(subset=['entry_date'])

df['entry_years'] = 2025 - df['entry_date'].dt.year
df = df.drop(columns=['entry_date'])

used_cols = [
    'title', 'hospital', 'city', 'good_rating', 'num_consults', 'num_flags',
    'visit_total', 'article_count', 'online_patients', 'num_reviews',
    'reply_quality_score', 'service_attitude_score', 'reply_speed_score',
    'entry_years', 'price_per_min'
]
df = df[used_cols].dropna()

for col in ['title', 'hospital', 'city']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['price_per_min'])
y = df['price_per_min']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 8))  #
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("医生价格影响因素（SHAP重要性条形图）", fontsize=16)
plt.tight_layout()  # 自动调整布局避免标题被切割
plt.savefig("医生价格_SHAP重要性条形图.png")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("医生价格影响因素（SHAP分布图）", fontsize=16)
plt.tight_layout()
plt.savefig("医生价格_SHAP分布图.png")
plt.close()
