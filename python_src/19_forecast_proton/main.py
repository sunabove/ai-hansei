# -*- coding: utf-8 -*-

print( " Hello..... ".center(50, "*") )

if 1 : # TODO 1000 import packages
    import logging
    import os
    import argparse
    import math
    import datetime
    import numpy as np
    import pandas as pd
    import time
    import csv
    import matplotlib.pyplot as plt
    import sqlite3 
    import pandas
    import pandasql as psql
    import pysqldf

    from PIL import Image

    import tensorflow as tf
    from tensorflow import keras

    from tensorflow.keras import callbacks 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Lambda  
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Activation
    from tensorflow.python.keras import backend as K

    import torch
    import torch.nn as nn

    from torch.optim.lr_scheduler import StepLR
    from torch.utils import data
    from torchvision import datasets, transforms

    from dataloader import data_loader
pass #-- import packages

# TODO: 1000 로그 패키지 설정
logging.basicConfig(format='%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)
LINE = "*"*100

# TODO 1001 argument parser
args = argparse.ArgumentParser()
args.add_argument("--lr", type=int, default=0.001)
args.add_argument("--cuda", type=bool, default=True)
args.add_argument("--num_epochs", type=int, default=300 )  #TODO 2011 훈련수 num ephocs set
args.add_argument("--print_iter", type=int, default=10)
args.add_argument("--model_name", type=str, default="model.pth")
args.add_argument("--prediction_file", type=str, default="prediction.txt")
args.add_argument("--batch", type=int, default=24)
args.add_argument("--mode", type=str, default="train")

config = args.parse_args()
# TODO 1001-- argument parse

mode = config.mode

# TODO 1002 dataset path 설정
ds_path = ds_path = os.path.join( './datasets/' )
log.info("DATASET_PATH : %s" % ds_path)
# -- dataset path 설정 

db_version = 6 # TODO 1002 데이터베이스 버전
db_file_name = "proton_%03d.sqlite" % db_version
conn = sqlite3.connect( db_file_name )
cur = conn.cursor()

# table creation after checking it.
def check_table_exists(tablename):
    cur = conn.cursor()

    cur.execute("""
        SELECT IFNULL( COUNT(*) , 0 )
        FROM sqlite_master 
        WHERE name = '{0}'
        """.format(tablename.replace('\'', '\'\'')))

    cnt = cur.fetchone()[0]

    cur.close()

    return cnt
pass

if not check_table_exists( "xray" ) : 
    #conn.execute( "drop table xray" )
    sql = '''
        create table if not exists xray( 
            phase varchar( 10 ) , 
            id integer ,
            time_tag timestamp,

            xs float, 
            xl float,

            dxs float default null,
            dxl float default null,

            primary key( phase, id)
        )
    '''
    log.info( sql )
    cur.execute( sql ) 
pass #-- xray

if not check_table_exists( "proton" ) : 
    #conn.execute( "drop table proton" )
    sql = '''
        create table if not exists proton( 
            phase varchar( 10 ) , 
            id integer ,
            time_tag timestamp,

            proton float,  

            dproton float default null,

            primary key( phase, id)
        )
    '''
    log.info( sql )
    cur.execute( sql ) 
pass #-- proton

if not check_table_exists( "epm" ) : 
    #conn.execute( "drop table em" )
    sql = '''
        create table if not exists epm( 
            phase varchar( 10 ) , 
            id integer ,
            time_tag timestamp,

            p1p float, 
            p2p float, 
            p3p float, 
            p4p float, 
            p5p float, 
            p6p float, 
            p7p float, 
            p8p float, 

            dp1p float default null,
            dp2p float default null, 
            dp3p float default null, 
            dp4p float default null, 
            dp5p float default null, 
            dp6p float default null, 
            dp7p float default null, 
            dp8p float default null,

            primary key( phase, id)
        )
    '''
    log.info( sql )
    cur.execute( sql )  
pass #-- epm

if not check_table_exists( "swe" ) : 
    #conn.execute( "drop table em" )
    sql = '''
        create table if not exists swe( 
            phase varchar( 10 ) , 
            id integer ,
            time_tag timestamp ,

            h_density float , 
            sw_h_speed float , 

            dh_density float default null, 
            dsw_h_speed float default nul,

            primary key( phase, id)
        )
    '''
    log.info( sql )
    cur.execute( sql )  
pass #-- epm
#-- table creation after checking it.

def table_row_count( table_name , phase ) :
    cur = conn.cursor()
    
    sql = "select IFNULL( count(*), 0 ) from %s where phase = ?" % (table_name)

    log.info( sql )

    cur.execute( sql, (phase,) )
    cnt = cur.fetchone()[0]

    cur.close()

    return cnt
pass # -- table row count

def import_csv_to_sqlite( phase ) :
    log.info( "Import %s csv data to sqlite ...." % phase )

    ds_list[ phase ] = {}
    ds = ds_list[ phase ]

    def update_time_tag( table_name , table_name_org ) :
        sql_org = "update %s set time_tag = replace( replace( replace( time_tag, 'T', ' ' ), 'Z', ' ' ), ':00.000', '' )"
        
        sql = sql_org % ( table_name )
        log.info( sql )
        conn.execute( sql )

        sql = sql_org % ( table_name_org )

        log.info( sql )
        conn.execute( sql )
    pass #-- update_time_tag

    def update_negative_data_as_null( table_name, col_name , phase ) :
        sql = "update %s set %s = NULL where %s = ? and phase = ?"
        sql = sql % ( table_name, col_name , col_name )

        log.info( sql )
        conn.execute( sql , (-100, phase,) )
    pass #-- update_negative_data_as_null

    def update_increment_value( table_name, col_name, phase ) :
        sql = '''
            update %s t2 
            set d%s = (     (
                                selecdt 
                                    ( t2.%s - t1.%s ) as ds ,
                                    ( round( ( julianday(t2.time_tag) - julianday(t1.time_tag) )*86400.0 ) as dt
                                from %s t1 
                                where t1.phase = ? 
                                    and t1.id < t2.id 
                                    and t1.%s is not null
                                limit 1 
                             ) 
                     )
            where t2.pahse = ? and t2.%s is not null
        '''
        sql = sql % ( table_name, col_name , col_name, col_name, table_name, phase, phase )

        log.info( sql )
        conn.execute( sql )
    pass #-- update_increment

    table_name = "xray"

    if table_row_count( table_name, phase )  < 1 :    
        log.info( "Import from %s" % table_name )

        df = ds[ table_name ] = pd.read_csv( os.path.join( ds_path, "%s/" % phase , "%s_%s.csv" % (phase, table_name ) ) )

        log.info( ds[ table_name ] ) 

        table_name_org = "%s_%s_org" % (table_name, phase)

        df.to_sql( table_name_org, conn, if_exists='replace', index=False)

        sql = "insert into %s( phase, id, time_tag, xs, xl ) select '%s', rowid, time_tag, xs, xl from %s"
        sql = sql % ( table_name, phase, table_name_org )

        log.info( sql )

        cur.execute( sql )

        update_time_tag( table_name, table_name_org )
        
        col_names = ( "xs" , "xl" )
        for col_name in col_names :
            update_negative_data_as_null( table_name, col_name, phase )
            update_increment_value( table_name, col_name, phase )
        pass

    pass #-- xray

    table_name = "proton"

    if table_row_count( table_name, phase )  < 1 :
        log.info( "Import from %s" % table_name )

        df = ds[ table_name ] = pd.read_csv( os.path.join( ds_path, "%s/" % phase , "%s_%s.csv" % ( phase, table_name ) ) )

        log.info( ds[ table_name ] ) 

        table_name_org = "%s_%s_org" % (table_name, phase)

        df.to_sql( table_name_org, conn, if_exists='replace', index=False)

        sql = "insert into %s( phase, id, time_tag, proton ) select '%s', rowid, time_tag, proton from %s"
        sql = sql % ( table_name, phase, table_name_org )

        log.info( sql )

        cur.execute( sql )

        update_time_tag( table_name, table_name_org )

        col_names = [ "proton" ]
        for col_name in col_names :
            update_negative_data_as_null( table_name, col_name, phase )
            update_increment_value( table_name, col_name, phase )
        pass
        
    pass #-- proton

    table_name = "epm" 
    if table_row_count( table_name, phase )  < 1 :
        log.info( "Import from %s" % table_name )

        df = ds[ table_name ] = pd.read_csv( os.path.join( ds_path, "%s/" % phase , "%s_AC_H1_%s.csv" % ( phase, table_name.upper() ) ) )

        log.info( ds[ table_name ] ) 

        table_name_org = "%s_%s_org" % (table_name, phase)

        df.to_sql( table_name_org, conn, if_exists='replace', index=False)

        sql = "insert into %s( phase, id, time_tag, p1p, p2p, p3p, p4p, p5p, p6p, p7p, p8p ) select '%s', rowid, * from %s"
        sql = sql % ( table_name, phase, table_name_org )

        log.info( sql )

        cur.execute( sql )

        update_time_tag( table_name, table_name_org )
        
        col_names = [ "p1p" , "p2p", "p3p", "p4p", "p5p", "p6p", "p7p", "p8p" ]
        for col_name in col_names :
            update_negative_data_as_null( table_name, col_name, phase )
            update_increment_value( table_name, col_name, phase )
        pass
    pass #-- epm

    table_name = "swe"
    if table_row_count( table_name, phase )  < 1 :
        log.info( "Import from %s" % table_name )

        df = ds[ table_name ] = pd.read_csv( os.path.join( ds_path, "%s/" % phase , "%s_AC_H0_%s.csv" % ( phase, table_name.upper() ) ) )

        log.info( ds[ table_name ] )

        table_name_org = "%s_%s_org" % (table_name, phase)

        df.to_sql( table_name_org, conn, if_exists='replace', index=False)

        sql = "insert into %s( phase, id, time_tag, h_density, sw_h_speed ) select '%s', rowid, * from %s"
        sql = sql % ( table_name, phase, table_name_org )

        log.info( sql )

        cur.execute( sql )

        update_time_tag( table_name, table_name_org )

        col_names = [ "h_density", "sw_h_speed" ]
        for col_name in col_names :
            update_negative_data_as_null( table_name, col_name, phase )
            update_increment_value( table_name, col_name, phase )
        pass

    pass #-- swe
pass #-- import csv to sqlite

ds_list = {} 

phases = ( "train" , "val", "test" )

for phase in phases :
    import_csv_to_sqlite( phase )

    conn.commit()
pass

sql = '''
    select t2.time_tag, 
    t2.xs - t1.xs as dxs ,
    t2.xl - t1.xl as dxl 
    from xray t2, xray t1
    where 1 
        and t2.phase = 'train' and t1.phase='train' 
        and t2.id = t1.id - 1
    order by t2.phase, t2.time_tag, t2.id
    limit 100
'''

log.info( sql )

df = pandas.read_sql_query( sql, conn)

log.info( df )

cur.close()

conn.close()

print( " Good bye. ".center(50, "*") )