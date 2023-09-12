import os
import ftputil
from dotenv import load_dotenv


# 載入.env設定檔
load_dotenv(dotenv_path='pg.env')

ftp_host = os.getenv('ftp_host')
ftp_user = os.getenv('ftp_user')
ftp_password = os.getenv('ftp_password')
remote_folder = os.getenv('remote_folder')  # FTP 伺服器中的檔案路徑
local_folder = os.getenv('local_folder')

# 檢查 local_folder 是否存在，如果不存在則創建它
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

def download_files_from_ftp(host, user, password, remote_folder, local_folder):
    try:
        with ftputil.FTPHost(host, user, password) as host:
            host.chdir(remote_folder)

            file_list = host.listdir(host.curdir)

            for filename in file_list:
                remote_filepath = f"{remote_folder}/{filename}"
                local_filepath = f"{local_folder}/{filename}"
                host.download(remote_filepath, local_filepath)

        print(f"已成功下載文件夾 '{remote_folder}' 中的檔案到資料夾 '{local_folder}'。")

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    download_files_from_ftp(ftp_host, ftp_user, ftp_password, remote_folder, local_folder)
