import os
import glob
import pandas as pd
from tqdm import tqdm  # 匯入tqdm庫，用於顯示進度條

def process_file(filename, output_file, first_file, chunksize=10**6):
    first_chunk = True
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # Extract model and variable from column name
        variable, model, scenario = chunk.columns[3].split('_')

        # Normalize column names and add model column
        chunk = chunk.rename(columns={chunk.columns[3]: model + '_' + variable})

        if first_file and first_chunk:
            chunk.to_csv(output_file, mode='w', index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file, mode='a', header=False, index=False)

# 取得指定目錄中所有csv檔案的清單
files = glob.glob(r'D:\TTCIP\SSP126\*.csv')

# 指定輸出檔案
output_file = r'D:\TTCIP\SSP126\cleaned_data.csv'

# 處理所有檔案
first_file = True
for f in tqdm(files):
    process_file(f, output_file, first_file)
    first_file = False
