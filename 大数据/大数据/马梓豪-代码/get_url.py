import requests
from bs4 import BeautifulSoup
import time
import pymysql
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',               # ✅ 已设置
    'password': '123456',         # ✅ 已设置
    'database': 'jd_data',
    'charset': 'utf8mb4'
}

province_urls = {
    '上海': '2_0-4-',
    '江苏': '12_0-4-',
    '浙江': '15_0-4-',
    '四川': '22_0-4-',
    '重庆': '4_0-4-',
    '山西': '6_0-4-',
    '山东': '13_0-4-',
    '河南': '7_0-4-',
    '河北': '5_0-4-'
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://cont.jd.com/',
    'Connection': 'keep-alive',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9'
}

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)
seen_urls = set()
doctor_info_list = []

for province, url_prefix in province_urls.items():
    page_num = 1
    print(f"\n 正在抓取【{province}】皮肤科医生信息...")

    while True:
        url = f'https://cont.jd.com/department/{url_prefix}{page_num}?secondDepartmentId=7300035&doctorTitleId=0&sortItem=1&doctorServiceTypeId=0'
        try:
            response = session.get(url, headers=headers, timeout=10)
            print(f" [{province}] 第 {page_num} 页状态码：{response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f' [{province}] 第 {page_num} 页请求异常：{e}')
            break

        if response.status_code != 200:
            print(f' [{province}] 第 {page_num} 页跳过（状态码：{response.status_code}）')
            break

        soup = BeautifulSoup(response.text, 'html.parser')


        name_tags = soup.find_all('h5', class_='name')
        if not name_tags:
            print(f' [{province}] 第 {page_num} 页无医生信息，结束。')
            break

        cards = soup.find_all('a', class_='item')
        for card in cards:
            href = card.get('href')
            name_tag = card.find('h5', class_='name')

            if not href or not name_tag:
                continue

            doctor_url = href if href.startswith('http') else 'https://cont.jd.com' + href
            doctor_name = name_tag.text.strip()

            if doctor_url in seen_urls:
                continue

            seen_urls.add(doctor_url)
            doctor_info_list.append((doctor_name, doctor_url, province))

        print(f' [{province}] 第 {page_num} 页抓取成功，累计医生数：{len(doctor_info_list)}')
        page_num += 1
        time.sleep(1.5)


print(f"\n 共准备插入 {len(doctor_info_list)} 条医生信息...")

try:
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("TRUNCATE TABLE doctor_internet")

    insert_sql = "INSERT INTO doctor_internet (name, url, province) VALUES (%s, %s, %s)"
    cursor.executemany(insert_sql, doctor_info_list)
    conn.commit()

    print(f" 成功插入 {cursor.rowcount} 条记录到 doctor_internet 表。")

except Exception as e:
    print(" 数据写入失败：", e)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()

print(" 所有任务完成。")