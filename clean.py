import pandas as pd

# 讀取Excel文件
df = pd.read_excel("C:/air_box/air_quality_data.xlsx")

# 將'time'列轉換為datetime對象並去除時區信息
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# 創建新的日期和時間列
df['date'] = df['time'].dt.date
df['clock'] = df['time'].dt.time

# 將'name'列分割成'building'和'class'兩列
df[['building', 'class']] = df['name'].str.split('_', expand=True)

# 印處理後的DataFrame的前幾行
print(df.head())

# 將處理後的DataFrame儲存到Excel文件，並將工作表名稱設置為'air_box'
df.to_excel("C:/air_box/air_quality_data_forBI.xlsx", index=False, sheet_name='air_box')

