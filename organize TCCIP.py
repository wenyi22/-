import pandas as pd
import os
import re
from tqdm import tqdm

folder_path = r"D:\TTCIP\最高溫\historical"
output_folder = r"D:\TTCIP\最高溫\historical\test"
output_file_pattern = os.path.join(output_folder, "AR6_統計降尺度_日資料_臺灣_{}_{}_{}.csv")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".csv") and filename != "output.csv"]

for i, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
    filename = os.path.basename(file_path)
    match = re.search(r'AR6_統計降尺度_日資料_臺灣_([\w-]+)_([\w-]+)_(\d{4})', filename)
    if not match:
        print(f'Filename does not match regex: {filename}')
        continue

    var_type = match.group(1)
    model_name = match.group(2)
    year = match.group(3)

    try:
        df = pd.read_csv(file_path)
        df.fillna(-99.9, inplace=True)
        df.columns = ['LON', 'LAT'] + pd.to_datetime(df.columns[2:], format='%Y%m%d', errors='coerce').tolist()
        df_melt = df.melt(id_vars=["LON", "LAT"], var_name="DATE", value_name=f"{model_name}_{var_type}") 

        output_file = output_file_pattern.format(var_type, model_name, year)
        df_melt.to_csv(output_file, index=False)
        
        print(f"資料已成功寫入csv檔案：{output_file}")
        
        del df
        del df_melt

    except Exception as e:
        print(f'Error processing file: {filename}, error: {e}')
