import pymysql
import requests
from bs4 import BeautifulSoup
import re
import csv

def load_start_id():
    try:
        with open("failed_doctor_ids.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    return int(row[0])
    except FileNotFoundError:
        return 1
    return 1

def write_failed_id(doctor_id):
    with open("failed_doctor_ids.csv", "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["doctor_id"])
        writer.writerow([doctor_id])


conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='jd_data',
    charset='utf8mb4'
)
cursor = conn.cursor()

cursor.execute("SELECT id, url FROM doctor_internet")
rows = cursor.fetchall()

start_doctor_id = load_start_id()
print(f"从 doctor_id >= {start_doctor_id} 开始爬取...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

insert_sql = """
    INSERT INTO patient_comments (
        doctor_id, username, comment_text, consult_type, comment_date,
        reply_quality, service_attitude, reply_speed, overall_star
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

for row_idx, row in enumerate(rows, 1):
    doctor_id, doctor_url = row

    if doctor_id < start_doctor_id:
        continue

    cursor.execute("SELECT COUNT(*) FROM patient_comments WHERE doctor_id = %s", (doctor_id,))
    if cursor.fetchone()[0] > 0:
        print(f"\n 医生 {doctor_id} 已写入过评论，跳过")
        continue

    match = re.search(r'/doctor/(\d+)', doctor_url)
    if not match:
        print(f"跳过：URL 无法提取 doctor_id：{doctor_url}")
        continue
    real_doctor_number = match.group(1)

    try:
        check_resp = requests.get(doctor_url, headers=headers, timeout=10)
        check_resp.encoding = check_resp.apparent_encoding
        check_soup = BeautifulSoup(check_resp.text, "html.parser")
        doctor_name_tag = check_soup.select_one("h1.name")
        if not doctor_name_tag:
            print(f" 被反爬！医生 {doctor_id} 页面无 h1.name，程序终止")
            write_failed_id(doctor_id)
            cursor.close()
            conn.close()
            exit(1)
    except Exception as e:
        print(f" 医生主页访问异常（doctor_id={doctor_id}）：{e}")
        write_failed_id(doctor_id)
        cursor.close()
        conn.close()
        exit(1)

    comment_url = f"https://cont.jd.com/dianping/{real_doctor_number}_1?diseaseLabelActive"
    print(f"\n=== 正在爬取医生 {doctor_id} 的评论 ===")
    print(f"访问地址：{comment_url}")

    try:
        response = requests.get(comment_url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("div.card")

        if not cards:
            cursor.execute(insert_sql, (
                doctor_id,
                None, None, None, None,
                None, None, None, None
            ))
            conn.commit()
            print(f"医生 {doctor_id} 无评论，写入空记录")
            continue

        for card in cards:
            name_tag = card.select_one(".name")
            detail_tag = card.select_one(".detail")
            type_tag = card.select_one(".type")
            time_tag = card.select_one(".time")
            quality_tag = card.select_one(".quality")
            star_tags = card.select("div.star")

            if not (name_tag and detail_tag and time_tag):
                continue

            username = name_tag.get_text(strip=True)
            raw_comment = detail_tag.get_text(strip=True)
            comment_text = raw_comment.replace("评价详情：", "").strip()
            if comment_text == "":
                comment_text = None

            consult_type = type_tag.get_text(strip=True).replace("问诊类型：", "") if type_tag else None
            comment_date = time_tag.get_text(strip=True)
            overall_star = len(star_tags)

            reply_quality = service_attitude = reply_speed = None
            if quality_tag:
                parts = [p.strip() for p in quality_tag.get_text(strip=True).split("｜")]
                for p in parts:
                    if "回复质量" in p:
                        reply_quality = p.split("：")[-1]
                    elif "服务态度" in p:
                        service_attitude = p.split("：")[-1]
                    elif "回复速度" in p:
                        reply_speed = p.split("：")[-1]

            cursor.execute(insert_sql, (
                doctor_id,
                username,
                comment_text,
                consult_type,
                comment_date,
                reply_quality,
                service_attitude,
                reply_speed,
                overall_star
            ))
            conn.commit()

            print(f"写入评论：{username} - {comment_date} - 内容：{comment_text if comment_text else 'NULL'}")

    except Exception as e:
        print(f"医生 {doctor_id} 评论页爬取失败：{e}")

cursor.close()
conn.close()
print("\n 所有医生评论处理完毕。")
