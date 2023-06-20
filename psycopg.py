import psycopg2






database = "Raw_Data"

# 建立與資料庫的連接
conn = psycopg2.connect(database="Raw_Data", user="postgres", password="1!qaz2@wsx", host="140.134.51.64", port="5432")
cursor = conn.cursor()

# 建立"wra_res"表的SQL指令
create_table_query = '''
    CREATE TABLE wra_res (
        res_id SERIAL PRIMARY KEY,
        res_RN TEXT,
        res_RI TEXT,
        res_RD TEXT,
        res_RT TEXT,
        res_OT TEXT,
        res_OD TEXT,
        res_OF TEXT,
        res_IV TEXT,
        res_FWL TEXT,
        res_EC TEXT,
        res_DSL TEXT,
        res_CF TEXT,
        res_CAR TEXT
    )
'''

# 執行建立表格的指令
cursor.execute(create_table_query)
conn.commit()

# 關閉資料庫連接
cursor.close()
conn.close()
