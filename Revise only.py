import psycopg2

# 建立与数据库的连接
conn = psycopg2.connect(database="Raw_Data", user="postgres", password="1!qaz2@wsx", host="140.134.51.64", port="5432")
cursor = conn.cursor()

# 移除唯一性约束
alter_query = '''
    ALTER TABLE wra_res
    DROP CONSTRAINT IF EXISTS unique_res_id
'''

# 执行修改表结构的指令
cursor.execute(alter_query)
conn.commit()

# 关闭数据库连接
cursor.close()
conn.close()
