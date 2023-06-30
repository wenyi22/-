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
csv_file_path = os.path.join(script_dir, "wra_res_water_regime.csv")
xlsx_file_path = os.path.join(script_dir, "wra_res_water_regime.xlsx")
translation_file_path = os.path.join(script_dir, "wra.xlsx")

url = "https://data.wra.gov.tw/Service/OpenData.aspx?format=csv&id=1602CA19-B224-4CC3-AA31-11B1B124530F"
# filename = "wra_res_water_regime.csv"


# 使用指定的憑證檔案
cert_file = os.path.join(os.path.dirname(sys.argv[0]), 'cacert.pem')
response = requests.get(url, verify=cert_file)

# 發送 GET 請求並獲取 CSV 資料
response = requests.get(url)
data = response.content.decode("utf-8")

# 儲存 CSV 資料到檔案
with open(csv_file_path, "w", encoding="utf-8") as file:
    file.write(data)

print("CSV 資料已獲取並儲存到", csv_file_path)

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

# 讀取資料
# df = pd.read_csv("wra_res_water_regime.csv")
df = pd.read_csv(csv_file_path)
# 讀取對照表檔案
translation_df = pd.read_excel(translation_file_path)
# translation_df = pd.read_excel("水庫對照表格.xlsx")

# 建立對照字典
translation_dict = pd.Series(translation_df.res_id.values, index=translation_df.res_number).to_dict()

# 進行對照轉換，並且將結果儲存到新的一列 res_id
df["res_id"] = df["ReservoirIdentifier"].map(translation_dict)

# 在 DataFrame 中新增 "res_name" 列，根据 "res_id" 列的值进行查找
df["res_name"] = df["res_id"].map(translation_df.set_index("res_id")["res_name"])

# 將 res_Date 列的值設為 "%Y-%m-%d"
df['res_date'] = pd.to_datetime(df['ObservationTime']).dt.strftime('%Y-%m-%d')

# 縮寫
abbreviations = {
    'AccumulateRainfallInCatchment': 'res_ARainfallIC',
    'DesiltingTunnelOutflow': 'res_DesTOutflow',
    'DrainageTunnelOutflow': 'res_DrainTOutflow',
    'EffectiveWaterStorageCapacity': 'res_EWaterStorageC',
    'InflowDischarge': 'res_InflowD',
    'ObservationTime': 'res_Time',
    'OthersOutflow': 'res_OOutflow',
    'PowerOutletOutflow': 'res_PowerOutflow',
    'PredeterminedCrossFlow': 'res_PCrossfloe',
    'PredeterminedOutflowTime': 'res_POutflowT',
    'ReservoirIdentifier': 'res_RID',
    'SpillwayOutflow': 'res_SOutflow',
    'StatusType': 'res_SType',
    'TotalOutflow': 'res_TotalOutflow',
    'WaterDraw': 'res_WDraw',
    'WaterLevel': 'res_WLevel'
}


df = df.rename(columns=abbreviations)

# 將 res_OT 列轉換為日期時間格式
df['res_Time'] = pd.to_datetime(df['res_Time']).dt.strftime('%H:%M:%S')

# df.to_excel("wra_res_water_regime.xlsx", index=False)
df.to_excel(xlsx_file_path, index=False)

# 读取Excel文件
# df = pd.read_excel("wra_res_water_regime.xlsx")
df = pd.read_excel(xlsx_file_path)

# 重新排列列的顺序
df = df[["res_id", "res_name", "res_RID", "res_date", "res_Time", "res_ARainfallIC", "res_DesTOutflow", "res_DrainTOutflow", "res_EWaterStorageC", "res_InflowD", "res_OOutflow", "res_PowerOutflow", "res_PCrossfloe", "res_POutflowT", "res_SOutflow", "res_SType", "res_TotalOutflow", "res_WDraw", "res_WLevel"]]

# 建立与数据库的连接
conn = pg8000.connect(database="Rawdata_WRA", user="postgres", password="password", host="host", port="5432")
cursor = conn.cursor()

df['res_SType'] = df['res_SType'].fillna(-9999).astype(int)

# 将整理后的数据逐行插入到"wra_res"表格中
for _, row in df.iterrows():
    # 使用前一行的有效值填充 NaN
    row = row.fillna(-9999)
    
    insert_query = '''
        INSERT INTO wra_res_water_regime (res_id, res_name, res_RID, res_date, res_Time, res_ARainfallIC, res_DesTOutflow, res_DrainTOutflow, res_EWaterStorageC, res_InflowD, res_OOutflow, res_PowerOutflow, res_PCrossfloe, res_POutflowT, res_SOutflow, res_SType, res_TotalOutflow, res_WDraw, res_WLevel)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    '''
    values = tuple(row)
    cursor.execute(insert_query, values)

# 提交事务并关闭连接
conn.commit()
cursor.close()
conn.close()