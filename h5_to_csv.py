import h5py
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import re

load_dotenv(dotenv_path='h5_to_csv.env')

output_folder = os.environ.get('OUTPUT_FOLDER')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 指定包含.h5文件的文件夹路径
folder_path = os.environ.get('FOLDER_PATH')

print(f'FOLDER_PATH: {folder_path}')
print(f'OUTPUT_FOLDER: {output_folder}')


def extract_date_str(filename):
    match = re.search(r'\d{8}T\d{6}', filename)
    if match:
        return match.group()
    return None


def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):
                yield (path, item)
            elif isinstance(item, h5py.Group):
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def process_h5_file(src_file):
    with h5py.File(src_file, 'r') as f:
        filename = os.path.basename(src_file)
        date_str = extract_date_str(filename)

        if date_str is None:
            print(f"無法正確讀取文件名稱：{filename}")
            return None

        print(f"Extracted date_str: {date_str}")
        date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")

        arr_lon = f['/Soil_Moisture_Retrieval_Data/longitude_centroid'][:]
        arr_lat = f['/Soil_Moisture_Retrieval_Data/latitude_centroid'][:]
        arr_sm = f['/Soil_Moisture_Retrieval_Data/soil_moisture'][:]

        df = pd.DataFrame({
            'lon': arr_lon,
            'lat': arr_lat,
            'sm': arr_sm,
            'date': date.date()
        })
        return df

if not os.path.exists(folder_path):
    print(f'找不到指定的包含.h5文件的文件夹路径：{folder_path}')
else:
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    for src_file in h5_files:
        df = process_h5_file(os.path.join(folder_path, src_file))
        if df is not None:
            output_file = os.path.join(output_folder, f'{src_file.split("_")[7][:8]}_smois.csv')
            df.to_csv(output_file, sep=',', encoding='ANSI', index=False)
            print(f'{src_file} processed and saved as {output_file}')
