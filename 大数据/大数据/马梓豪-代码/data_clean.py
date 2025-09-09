import pandas as pd
from sqlalchemy import create_engine
import numpy as np

user = 'root'
password = '123456'
host = 'localhost'
port = 3306
database = 'jd_data'

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4")

doctor_info = pd.read_sql("SELECT * FROM doctor_info", con=engine)
doctor_price = pd.read_sql("SELECT * FROM doctor_price_info", con=engine)
patient_comments = pd.read_sql("SELECT * FROM patient_comments", con=engine)

doctor_info.dropna(subset=["name", "hospital", "department"], inplace=True)
doctor_info.fillna({
    'reply_quality_score': doctor_info['reply_quality_score'].mean(),
    'service_attitude_score': doctor_info['service_attitude_score'].mean(),
    'reply_speed_score': doctor_info['reply_speed_score'].mean()
}, inplace=True)
doctor_info['entry_date'] = pd.to_datetime(doctor_info['entry_date'], errors='coerce')
doctor_info['good_rating'] = pd.to_numeric(doctor_info['good_rating'], errors='coerce').fillna(0)
doctor_info.drop_duplicates(subset=['id'], inplace=True)

doctor_price['type'] = doctor_price['type'].fillna('未知')
doctor_price[['price', 'price_per_min']] = doctor_price[['price', 'price_per_min']].fillna(0)
doctor_price['price'] = pd.to_numeric(doctor_price['price'], errors='coerce').fillna(0)
doctor_price['price_per_min'] = pd.to_numeric(doctor_price['price_per_min'], errors='coerce').fillna(0)

patient_comments.dropna(subset=['comment_text'], inplace=True)
for col in ['reply_quality', 'service_attitude', 'reply_speed']:
    patient_comments[col] = patient_comments[col].replace('', np.nan).fillna('NULL')
patient_comments['comment_date'] = pd.to_datetime(patient_comments['comment_date'], errors='coerce')

doctor_all = pd.merge(
    doctor_info,
    doctor_price.groupby('doctor_id').agg({
        'price': 'mean',
        'price_per_min': 'mean'
    }).reset_index(),
    left_on='id',
    right_on='doctor_id',
    how='left'
)

comment_agg = patient_comments.groupby('doctor_id').agg({
    'id': 'count',
    'overall_star': 'mean'
}).rename(columns={'id': 'num_comments', 'overall_star': 'avg_star'}).reset_index()

doctor_all = pd.merge(doctor_all, comment_agg, left_on='id', right_on='doctor_id', how='left')
doctor_all[['num_comments', 'avg_star']] = doctor_all[['num_comments', 'avg_star']].fillna(0)

doctor_all.to_csv("cleaned_doctor_data.csv", index=False)
patient_comments.to_csv("cleaned_comments_data.csv", index=False)

"已保存为 cleaned_doctor_data.csv 和 cleaned_comments_data.csv，可用于后续分析。"
