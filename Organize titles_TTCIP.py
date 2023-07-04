import os
import csv

# 資料夾路徑
folder_path = r"D:\TTCIP\SSP126"

# 建立一個集合來存放標題名稱
header_names = set()

# 瀏覽資料夾內的所有檔案
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            headers = next(csv_reader)
            if len(headers) > 3:
                fourth_header = headers[3]
                header_names.add(fourth_header)

# 去除重複的標題
header_names = list(header_names)

# 將標題名稱存入同一個 TXT 檔案
output_file = r"D:\TTCIP\SSP126\header_names.txt"

with open(output_file, "w", encoding="utf-8") as txt_file:
    txt_file.write("\n".join(header_names))

# 開啟 header_names.txt 檔案
with open(output_file, "r", encoding="utf-8") as txt_file:
    lines = txt_file.readlines()

# 在每一行的末尾加上空格、"FLOAT" 和逗號
lines_with_float = [line.rstrip().replace("-", "_") + " FLOAT," for line in lines]

# 將修改後的內容寫回檔案
with open(output_file, "w", encoding="utf-8") as txt_file:
    txt_file.writelines(lines_with_float)
