import os
from ftplib import FTP
from dotenv import load_dotenv

# 載入.env設定檔
load_dotenv(dotenv_path='ftp.env')

script_dir = os.path.dirname(os.path.abspath(__file__))

ftp_host = os.getenv('ftp_host')
ftp_user = os.getenv('ftp_user')
ftp_password = os.getenv('ftp_password')
download_folder = os.getenv('remote_folder')  
output_pathfile = os.path.join(script_dir, os.getenv('output_pathfile'))

def download_files_from_ftp(host, user, password, folder_name, output_path):
    try:
        ftp = FTP(host)
        ftp.login(user=user, passwd=password)
        ftp.encoding = "gbk"

        # 移動到特定文件夾
        ftp.cwd(folder_name)

        # 取得檔案列表
        file_list = []
        ftp.dir(file_list.append)
        
        # 下載每個檔案
        for file_info in file_list:
            filename = file_info.split()[-1]
            full_path = os.path.join(output_path, filename)
            print(f"正在下載: {filename} 到 {full_path}")
            with open(full_path, "wb") as file:
                ftp.retrbinary("RETR " + filename, file.write)
        
        # 關閉
        ftp.quit()
        print(f"已成功下載文件夾 '{folder_name}' 中的檔案。")

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    if not os.path.exists(output_pathfile):
        os.makedirs(output_pathfile)
        
    download_files_from_ftp(ftp_host, ftp_user, ftp_password, download_folder, output_pathfile)