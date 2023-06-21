import psycopg2

# 建立與資料庫的連接
conn = psycopg2.connect(database="Raw_Data", user="postgres", password="1!qaz2@wsx", host="140.134.51.64", port="5432")
cursor = conn.cursor()

# 修改表格的欄位名稱
# alter_query1 = '''
#     ALTER TABLE wra_res
#     RENAME COLUMN res_rid TO res_Rid;
# '''
# cursor.execute(alter_query1)

alter_query2 = '''
    ALTER TABLE wra_res
    RENAME COLUMN res_rn TO res_name;
'''
cursor.execute(alter_query2)

# 提交交易並關閉連接
conn.commit()
cursor.close()
conn.close()
