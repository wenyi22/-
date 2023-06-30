import sys
import os
import requests
import csv
import pandas as pd
import numpy as np
import re
import pg8000
import certifi
import shutil
import errno
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), 'cacert.pem')

# 定义插件目录
plugins_dir = 'C:/Users/rex/Anaconda3/Lib/site-packages/PyQt5/Qt5/plugins'

# 确保插件目录存在
if not os.path.exists(plugins_dir):
    raise FileNotFoundError(f"PyQt5 plugins directory not found: {plugins_dir}")

# 添加插件目录到 datas
datas = [(plugins_dir, 'PyQt5/Qt/plugins')]

# 獲取所在目錄
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))



# 文件路徑
csv_file_path = os.path.join(script_dir, "wra_res_history_daily.csv")
xlsx_file_path = os.path.join(script_dir, "wra_res_history_daily.xlsx")
translation_file_path = os.path.join(script_dir, "wra.xlsx")

url = "https://data.wra.gov.tw/Service/OpenData.aspx?format=csv&id=50C8256D-30C5-4B8D-9B84-2E14D5C6DF71"
# filename = r"C:\Users\rex\Desktop\水庫水情\test\wra_res_history_daily.csv"

# 使用指定的憑證檔案
cert_file = os.path.join(os.path.dirname(sys.argv[0]), 'cacert.pem')
response = requests.get(url, verify=cert_file)

# 發送 GET 請求並獲取 CSV 資料
response = requests.get(url)
data = response.content.decode("utf-8")

# 儲存 CSV 資料到檔案
with open(csv_file_path, "w", encoding="utf-8") as file:
    file.write(data)

print("CSV 数据已获取并保存到", csv_file_path)

# 讀取 CSV 檔案並整理資料到正確的欄位
with open(csv_file_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    rows = list(reader)  # 將每一行轉換成列表

# 在原始檔案中寫入整理後的資料
with open(csv_file_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    # 寫入標頭行
    writer.writerow(rows[0])

    # 處理每一行資料（從第二行開始）
    for row in rows[1:]:
        # 檢查欄位數量是否符合預期
        if len(row) < 4:
            continue  # 跳過欄位數量不正確的行

        # 獲取欄位資料
        date = row[0]
        reservoir_name = row[1]
        water_level = row[2]
        capacity = row[3]

        # 將整理後的資料寫入檔案
        writer.writerow(row)

print("資料已整理並寫回原始檔案")


# 讀取 csv 檔
# df = pd.read_csv(r"C:\Users\rex\Desktop\水庫水情\test\wra_res_history_daily.csv")
df = pd.read_csv(csv_file_path)
# 讀取對照表檔案
# translation_df = pd.read_excel(r"C:\Users\rex\Desktop\水庫水情\水庫對照表格.xlsx")
translation_df = pd.read_excel(translation_file_path)
# 建立對照字典
translation_dict = pd.Series(translation_df.res_id.values, index=translation_df.res_name).to_dict()

# 進行對照轉換，並且將結果儲存到新的一列 res_id
df["res_id"] = df["ReservoirName"].map(translation_dict)


# 複製 res_id 這一列並命名為 res_id_copy
df["res_id_copy"] = df["res_id"]

# 將 res_id 中的空字符串替換成 NaN，以方便使用 fillna 函數
df['res_id'].replace('', np.nan, inplace=True)

# 使用 res_id_copy 的值來填補 res_id 的 NaN
df['res_id'].fillna(df['res_id_copy'], inplace=True)

# # 將 res_id_copy 這一列的所有值變成空值
# df['res_id_copy'] = np.nan

#縮寫
abbreviations = {
    'CatchmentAreaRainfall': 'res_CAreaRainfall',
    'CrossFlow': 'res_CrossFlow',
    'DeadStorageLevel': 'res_DStorageL',
    'EffectiveCapacity': 'res_ECapacity',
    'FullWaterLevel': 'res_FullWL',
    'InflowVolume': 'res_InflowV',
    'Outflow': 'res_Outflow',
    'OutflowDischarge': 'res_OutflowD',
    'OutflowTotal': 'res_OutflowT',
    'RecordTime': 'res_Time',
    'RegulatoryDischarge': 'res_RegulatoryD',
    'ReservoirIdentifier': 'res_RID',
    'ReservoirName': 'res_name',
    'res_id': 'res_id',
    'res_id_copy': 'res_id_copy'
}

df = df.rename(columns=abbreviations)

# 將 res_RT 列轉換為日期時間格式
df['res_Time'] = pd.to_datetime(df['res_Time'])

# 將日期格式轉換為 "YYYY-MM-DD" 格式
df['res_Time'] = df['res_Time'].dt.strftime('%Y-%m-%d')

# 將 res_id 的值存儲到新的一列 res_id_copy
df['res_id_copy'] = df['res_id']

# 將 res_RN 的值轉換為沒有數字的形式
df['res_name'] = df['res_name'].apply(lambda x: re.sub(r'\d+', '', x))

# 重新進行對照轉換，並將結果存儲到 res_id
df['res_id'] = df['res_name'].map(translation_dict)

# 刪除暫存的 res_id_copy 列
df.drop('res_id_copy', axis=1, inplace=True)

df.to_excel(xlsx_file_path, index=False)
# df.to_excel(r"C:\Users\rex\Desktop\水庫水情\test\wra_res_history_daily.xlsx", index=False)

# 建立與資料庫的連接
conn = pg8000.connect(database="Rawdata_WRA", user="postgres", password="password", host="host", port="5432")
cursor = conn.cursor()

# 讀取整理後的檔案
# df = pd.read_excel(r"C:\Users\rex\Desktop\水庫水情\test\wra_res_history_daily.xlsx")
df = pd.read_excel(xlsx_file_path)
# 將整理後的資料逐行插入到""表格中
for _, row in df.iterrows():
    # 使用前一個有效值填充 NaN
    row = row.fillna(-9999)
    
    insert_query = '''
        INSERT INTO wra_res_history_daily (res_CAreaRainfall, res_CrossFlow, res_DStorageL, res_ECapacity, res_FullWL, res_InflowV, res_Outflow, res_OutflowD, res_OutflowT, res_Time, res_RegulatoryD, res_RId, res_name, res_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    '''
    values = tuple(row)
    cursor.execute(insert_query, values)


# 提交交易並關閉連接
conn.commit()
cursor.close()
conn.close()