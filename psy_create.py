import psycopg2

# database = "Rawdata_WRA"

# # 建立与数据库的连接
# conn = psycopg2.connect(database="Rawdata_WRA", user="postgres", password="password", host="host", port="5432")

# cursor = conn.cursor()
# create_table_query = '''
#     CREATE TABLE wra_res_water_regime (
#         res_id INTEGER,
#         res_name VARCHAR(255),
#         res_RID INTEGER,
#         res_date DATE,
#         res_Time TIME,
#         res_ARainfallIC DOUBLE PRECISION,
#         res_DesTOutflow DOUBLE PRECISION,
#         res_DrainTOutflow DOUBLE PRECISION,
#         res_EWaterStorageC DOUBLE PRECISION,
#         res_InflowD DOUBLE PRECISION,
#         res_OOutflow DOUBLE PRECISION,
#         res_PowerOutflow DOUBLE PRECISION,
#         res_PCrossfloe DOUBLE PRECISION,
#         res_POutflowT DOUBLE PRECISION,
#         res_SOutflow DOUBLE PRECISION,
#         res_SType INTEGER,
#         res_TotalOutflow DOUBLE PRECISION,
#         res_WDraw DOUBLE PRECISION,
#         res_WLevel DOUBLE PRECISION
#     )
# '''

# # 执行建立表格的指令
# cursor.execute(create_table_query)
# conn.commit()

# # 关闭数据库连接
# cursor.close()
# conn.close()


database = "Rawdata_WRA"

# 建立與資料庫的連接
conn = psycopg2.connect(database="Rawdata_WRA", user="postgres", password="password", host="host", port="5432")


cursor = conn.cursor()
create_table_query = '''
    CREATE TABLE wra_res_history_daily (
        res_id INTEGER,
        res_name VARCHAR(255),        
        res_RId INTEGER,
        res_CAreaRainfall DOUBLE PRECISION,
        res_CrossFlow DOUBLE PRECISION,
        res_DStorageL DOUBLE PRECISION,
        res_ECapacity DOUBLE PRECISION,
        res_FullWL DOUBLE PRECISION,
        res_InflowV DOUBLE PRECISION,
        res_Outflow DOUBLE PRECISION,
        res_OutflowD DOUBLE PRECISION,
        res_OutflowT DOUBLE PRECISION,
        res_Time TEXT,
        res_RegulatoryD DOUBLE PRECISION
    )
'''

# 執行建立表格的指令
cursor.execute(create_table_query)
conn.commit()

# 關閉資料庫連接
cursor.close()
conn.close()

