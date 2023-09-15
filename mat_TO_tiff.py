from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from ftplib import FTP
from dotenv import load_dotenv

# 載入.env設定檔
load_dotenv(dotenv_path='ftp.env')

# Get the output path from the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_pathfile = os.path.join(script_dir, os.getenv('output_pathfile'))

# 建立你的 .mat 文件的完整路徑
picture_folder = os.path.join(script_dir, 'picture')
if not os.path.exists(picture_folder):
    os.makedirs(picture_folder)

# 載入 .mat 文件
for mat_filename in os.listdir(output_pathfile):
    if mat_filename.endswith('.mat'):
        mat_filepath = os.path.join(output_pathfile, mat_filename)
        mat_data = loadmat(mat_filepath)

        data_b01_aod = mat_data['Data_B01']['AOD'][0,0]
        data_b02_aod = mat_data['Data_B02']['AOD'][0,0]
        data_b03_aod = mat_data['Data_B03']['AOD'][0,0]

        data_b01_aod = np.nan_to_num(data_b01_aod, nan=0)
        data_b02_aod = np.nan_to_num(data_b02_aod, nan=0)
        data_b03_aod = np.nan_to_num(data_b03_aod, nan=0)

        data_b01_norm = ((data_b01_aod - data_b01_aod.min()) / (data_b01_aod.max() - data_b01_aod.min()) * 255).astype(np.uint8)
        data_b02_norm = ((data_b02_aod - data_b02_aod.min()) / (data_b02_aod.max() - data_b02_aod.min()) * 255).astype(np.uint8)
        data_b03_norm = ((data_b03_aod - data_b03_aod.min()) / (data_b03_aod.max() - data_b03_aod.min()) * 255).astype(np.uint8)

        img_b01 = Image.fromarray(data_b01_norm)
        img_b02 = Image.fromarray(data_b02_norm)
        img_b03 = Image.fromarray(data_b03_norm)

        # Save the images as a multi-page tiff
        tiff_filepath = os.path.join(picture_folder, mat_filename.replace('.mat', '_multi.tiff'))
        img_b01.save(tiff_filepath, save_all=True, append_images=[img_b02, img_b03])

        print(f"The multi-page TIFF file has been saved at: {tiff_filepath}")