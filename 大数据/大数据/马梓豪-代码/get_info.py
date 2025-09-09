import pymysql
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import csv
import os

db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'jd_data',
    'charset': 'utf8mb4'
}

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_doctor_ids.csv")
if not os.path.exists(csv_path):
    print(" 没有找到 failed_doctor_ids.csv 文件，请先初始化生成！")
    exit()

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    doctor_ids_to_retry = [int(row['doctor_id']) for row in reader]

conn = pymysql.connect(**db_config)
cursor = conn.cursor()

sql = "SELECT id, name, province, url FROM doctor_internet WHERE id IN %s"
cursor.execute(sql, (doctor_ids_to_retry,))
doctors = cursor.fetchall()


failed_ids = []
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://cont.jd.com/'
}

def safe_extract_number(text):
    if not text or '-' in text:
        return None
    match = re.search(r'\d+', text.replace(',', ''))
    return int(match.group()) if match else None

def safe_extract_percent(text):
    if not text or '-' in text:
        return None
    return float(text.strip('%')) if text.strip('%').replace('.', '').isdigit() else None

for doctor_id_local, name, province, url in doctors:
    cursor.execute("SELECT 1 FROM doctor_info WHERE id = %s", (doctor_id_local,))
    if cursor.fetchone():
        print(f" 已存在 doctor_info，跳过 ID: {doctor_id_local}")
        continue

    match = re.search(r'/doctor/(\d+)', url)
    if not match:
        failed_ids.append(doctor_id_local)
        continue

    doctor_id = match.group(1)
    intro_url = f"https://cont.jd.com/doctor/jianjie/{doctor_id}"
    review_url = f"https://cont.jd.com/dianping/{doctor_id}"

    try:
        resp = requests.get(intro_url, headers=headers, timeout=10)
        print(f" 请求 {intro_url} 返回状态码: {resp.status_code}")
        if resp.status_code != 200 or "您访问的页面不存在" in resp.text:
            raise Exception("伪造页面或跳转")

        soup = BeautifulSoup(resp.text, 'html.parser')
        def get_text(selector):
            tag = soup.select_one(selector)
            return tag.text.strip() if tag else None

        title = get_text('.yishengtitle .title')
        hospital = get_text('h3.alias a')
        if not title or not hospital:
            raise Exception("缺少关键信息，疑似反爬页面")

        alias_a_tags = soup.select('h3.alias a')
        department = alias_a_tags[-1].text.strip() if len(alias_a_tags) >= 2 else ''
        good_rating = safe_extract_percent(get_text('.yishengxiangqing .stat .detail:nth-of-type(1) .number'))
        num_consults = safe_extract_number(get_text('.yishengxiangqing .stat .detail:nth-of-type(2) .number'))
        num_flags = safe_extract_number(get_text('.yishengxiangqing .stat .detail:nth-of-type(3) .number'))
        visit_total = safe_extract_number(get_text('.hospitalDetailRow:nth-of-type(1) .hospitalDetailRowVal'))
        article_count = safe_extract_number(get_text('.hospitalDetailRow:nth-of-type(2) .hospitalDetailRowVal'))
        online_patients = safe_extract_number(get_text('.hospitalDetailRow:nth-of-type(3) .hospitalDetailRowVal'))
        num_reviews = safe_extract_number(get_text('.hospitalDetailRow:nth-of-type(4) .hospitalDetailRowVal'))
        entry_date_text = get_text('.hospitalDetailRow:nth-of-type(6) .hospitalDetailRowVal')
        entry_date = None
        try:
            entry_date = datetime.strptime(entry_date_text, "%Y-%m-%d %H:%M:%S") if entry_date_text else None
        except:
            pass

        speciality = get_text('.jieshaocontent .textholder')
        bio_tags = soup.select('.jieshaocontent .textholder')
        bio = bio_tags[1].text.strip() if len(bio_tags) > 1 else None

        # 获取评分页面
        try:
            review_resp = requests.get(review_url, headers=headers, timeout=10)
            review_soup = BeautifulSoup(review_resp.text, 'html.parser')

            def extract_score(title_kw):
                title_divs = review_soup.select('div.title')
                for title_div in title_divs:
                    if title_kw in title_div.text:
                        content = title_div.find_next_sibling('div', class_='content')
                        if content:
                            span = content.find('span')
                            return float(span.text.strip()) if span else None
                return None

            reply_quality_score = extract_score("回复质量")
            service_attitude_score = extract_score("服务态度")
            reply_speed_score = extract_score("回复速度")
        except:
            reply_quality_score = service_attitude_score = reply_speed_score = None

        cursor.execute("""
            INSERT INTO doctor_info (
                id, name, title, hospital, department, city,
                good_rating, num_consults, num_flags,
                speciality, bio, visit_total, article_count,
                online_patients, num_reviews, entry_date,
                reply_quality_score, service_attitude_score, reply_speed_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            doctor_id_local, name, title, hospital, department, province,
            good_rating, num_consults, num_flags, speciality, bio,
            visit_total, article_count, online_patients, num_reviews,
            entry_date, reply_quality_score, service_attitude_score, reply_speed_score
        ))
        conn.commit()
        print(f" 插入 doctor_info 成功 ID: {doctor_id_local}")

        cursor.execute("SELECT 1 FROM doctor_price_info WHERE doctor_id = %s LIMIT 1", (doctor_id_local,))
        if not cursor.fetchone():
            service_items = soup.select('.fuwuitemBox')
            for item in service_items:
                type_tag = item.select_one('.title')
                price_tag = item.select_one('.highlight')
                duration_tag = item.select_one('.time')
                service_type = type_tag.text.strip() if type_tag else None
                try:
                    price = float(price_tag.text.strip().replace('¥', '').replace('￥', '')) if price_tag else None
                except:
                    price = None
                duration = duration_tag.text.strip().replace('/', '') if duration_tag else None

                minutes = None
                if duration:
                    if '分钟' in duration:
                        minutes = int(re.search(r'\d+', duration).group())
                    elif '小时' in duration:
                        minutes = int(re.search(r'\d+', duration).group()) * 60
                    elif '天' in duration:
                        minutes = int(re.search(r'\d+', duration).group()) * 1440
                    elif '周' in duration:
                        minutes = int(re.search(r'\d+', duration).group()) * 7 * 1440
                    elif '月' in duration:
                        minutes = int(re.search(r'\d+', duration).group()) * 30 * 1440
                    elif '年' in duration:
                        minutes = int(re.search(r'\d+', duration).group()) * 365 * 1440

                price_per_min = round(price / minutes, 2) if price and minutes else None
                cursor.execute("""
                    INSERT INTO doctor_price_info (doctor_id, type, price, unit_duration, price_per_min)
                    VALUES (%s, %s, %s, %s, %s)
                """, (doctor_id_local, service_type, price, duration, price_per_min))
            conn.commit()
            print(f" 插入 doctor_price_info 成功 ID: {doctor_id_local}")
        else:
            print(f" doctor_price_info 已存在 ID: {doctor_id_local}")

    except Exception as e:
        print(f" 处理失败 ID: {doctor_id_local}，错误：{e}")
        failed_ids.append(doctor_id_local)

with open(csv_path, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['doctor_id'])
    for fid in failed_ids:
        writer.writerow([fid])

print(f" 本轮结束，失败 ID 已更新写入：{csv_path}")
cursor.close()
conn.close()