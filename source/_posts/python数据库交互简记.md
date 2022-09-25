---
title: python数据库交互简记
date: 2019-07-06
tags: 数据库
---

&emsp;&emsp;这里记录sqlite，mysql，SqlServer，以及oracle这几种数据库的导入导出和部分交互。
最基本的思路是使用sqlalchemy/cx_Oracle，以及针对dataframe使用pandas的read_sql，to_sql。
<!--more-->

```python
import pandas as pd
from sqlalchemy import create_engine

# 建立连接
# echo=True：打开调试；charset：指定链接编码；<name>:<password>：视数据库验证方式可否缺省
engine = create_engine(r'sqlite:///<path><name>.sqlite3', echo=True)  # sqlite
engine = create_engine("mysql://<name>:<password>@<ip>/db?charset=utf8")  # mysql
engine = create_engine('mssql+pymssql://<ip>/db?')  # SqlServer

# 数据读取
df = pd.read_sql('<SelectSQL>', engine)

# 数据导入，这里非常要注意if_exists（表存在时的处理方式）！！慎用replace他会重建表结构，一般append就好。
df.to_sql('<TableName>', engine, if_exists='append', index=False)

# 执行一句sql
conn = engine.connect()
conn.execute('<ExecSQL>')
conn.close()

# 执行SqlServer存储过程
import pymssql

conn = pymssql.connect(host='<ip>', database='<db>', user='<usr>', password='<pwd>')
cursor = conn.cursor()
sql_exec = """
exec ProcedureName
@ArgName = ArgValue"""
cursor.execute(sql_exec)
conn.commit()
cursor.close()
conn.close()

# oracle创建引擎需要用到SID或者附注tnsnames中链接的完整字段，且我在to_sql到他人schema中总报权限不足，
# 这里更推荐使用cx_Oracle以ServiceName直连：
import cx_Oracle

conn = cx_Oracle.connect('<name>/<password>@<ip>/<ServiceName>')

# 数据读取
df = pd.read_sql('<SelectSQL>', conn)


# 数据导入
def insert_df(df, table):
    """向oracle数据库插dataframe(pd.to_sql)"""
    conn = cx_Oracle.connect('<name>/<password>@<ip>/<ServiceName>')
    sql = "insert into " + table + "("
    for col in df.columns:
        sql += str(col) + ','
    sql = sql[:-1]
    sql += ') values(:'
    for col in df.columns:
        sql += str(col) + ',:'
    sql = sql[:-2]
    sql += ')'
    rec = df.to_json(orient='records', force_ascii=False)
    rec = rec.replace("\\", "")
    cursor = conn.cursor()
    cursor.prepare(sql)
    cursor.executemany(None, eval(rec))
    conn.commit()
    cursor.close()
    conn.close()


# 执行一句sql
def execute_sql(sql):
    """oracle,执行一句sql"""
    conn = cx_Oracle.connect('<name>/<password>@<ip>/<ServiceName>')
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


# 执行oracle存储过程
conn = cx_Oracle.connect('<name>/<password>@<ip>/<ServiceName>')
cursor = conn.cursor()
cursor.callproc("<schema.ProcedureName>", [ < ArgValues >])
cursor.close()
conn.close()
```
