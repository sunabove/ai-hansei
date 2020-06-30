# -*- coding: utf-8 -*-

from sqlite3 import connect
from pandas 
import pandasql
import pysqldf
import numpy

# CREATE AN IN-MEMORY SQLITE DB
con = connect(":memory:")
cur = con.cursor()
cur.execute("attach 'my.db' as filedb")
cur.execute("create table df as select * from filedb.hflights")
cur.execute("detach filedb")
 
# IMPORT SQLITE TABLE INTO PANDAS DF
df = pandas.read_sql_query("select * from df", con)
 
# WRITE QUERIES
sql01 = "select * from df where DayofWeek = 1 and Dest = 'CVG';"
sql02 = "select DayofWeek, AVG(ArrTime) from df group by DayofWeek;"
sql03 = "select DayofWeek, median(ArrTime) from df group by DayofWeek;"
 
# SELECTION:
# 1. PANDASQL
t11 = pandasql.sqldf(sql01, globals())
# 2. PYSQLDF
t12 = pysqldf.SQLDF(globals()).execute(sql01)
# 3. GENERIC SQLITE CONNECTION
t13 = read_sql_query(sql01, con)
 
# AGGREGATION:
# 1. PANDASQL
t21 = pandasql.sqldf(sql02, globals())
# 2. PYSQLDF
t22 = pysqldf.SQLDF(globals()).execute(sql02)
# 3. GENERIC SQLITE CONNECTION
t23 = read_sql_query(sql02, con)
 
# DEFINING A NEW FUNCTION:
# DEFINE A FUNCTION NOT SUPPORTED IN SQLITE
class median(object):
  def __init__(self):
    self.a = []
  def step(self, x):
    self.a.append(x)
  def finalize(self):
    return numpy.median(self.a)
 
# 1. PYSQLDF
udafs = {"median": median}
t31 = pysqldf.SQLDF(globals(), udafs = udafs).execute(sql03)
# 2 GENERIC SQLITE CONNECTION
con.create_aggregate("median", 1, median)
t32 = read_sql_query(sql03, con)