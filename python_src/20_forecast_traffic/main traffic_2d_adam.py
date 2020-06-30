# -*- coding: utf-8 -*-

import logging 
import os
import argparse
import math
import datetime
import numpy as np
import pandas as pd
import time
import csv
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda  
from tensorflow.keras.layers import LSTM
from tensorflow.keras import callbacks 
from tensorflow.python.keras import backend as K

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import datasets, transforms

from dataloader import data_loader
from evaluation import evaluation_metrics

# TODO: 1002 로그 패키지 임포트

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

log = logging.getLogger(__name__)

LINE = "*"*100

'''
DataLoader
'''

# TODO: 1007 Time Series Data Set
class TimeSeriesDataset(data.Dataset):
    def __init__(self, root, phase, debug = 1 ):
        file_name = phase
        fn = file_name

        mode = phase

        file_path = os.path.join(root, fn + '.csv')

        x_list = []
        y_list = []

        self.max_y = 0 
        self.min_data_list_size = 4 #TODO 2005 min data list 
            
        f = open( file_path, 'r', encoding='utf-8-sig' ) 
        # skip two lines
        f.readline()
        f.readline()

        # Train Dataset(2020. 01. 01 ~ 05. 01) 중 3월 30일은 수집 되지 않았음

        idx = 0 
        max_idx = 0  #TODO 1000 max idx

        min_data_list_size = self.min_data_list_size

        data_list = [ ]

        ymdh_prev = None  

        data_distcontinuous_cnt = 0 

        _999_encountered = 1 #-999 means predition data

        if mode == "test" :
            _999_encountered = 0 
        pass

        # 한 줄 씩 읽기
        for line in f.readlines():
            if max_idx and max_idx == idx :
                log.info( "**** max idx encounterd. %s" % max_idx )
                break
            pass
            
            row = np.array( line.strip().split(',') )

            ymd = int( row[0] )
            hour = int( row[1] )

            ymdh = datetime.datetime.strptime( "%04d %02d" % (ymd, hour), '%Y%m%d %H')

            data = None

            data = np.asfarray( row[2:], float )

            max_y = max( data )

            if max_y > self.max_y :
                self.max_y = max_y
            pass

            data_list_clear = 0 

            if ymdh_prev is None :
                pass
            else : 
                duration = ymdh - ymdh_prev 
                
                debug and log.info( "duration secons = %s" % duration.seconds )
                
                if duration.seconds != 3600 :
                    data_list_clear = 1 
                pass
            pass

            if data_list_clear : 
                # 시간차가 1시간이 아닐 경우, 데이터 목록을 재구성한다.
                data_list.clear()

                data_distcontinuous_cnt += 1 

                if debug : 
                    log.info( "[%s][%04d] %s" % (fn, idx, LINE) )
                    log.info( "[%s][%04d] hour is not continuous. ymdh = %s, ymdh_prev = %s" % (fn, idx, ymdh, ymdh_prev ) )
                    log.info( "[%s][%04d] %s" % (fn, idx, LINE) )
                pass
            pass

            debug and log.info( "[%s][%04d] %08d, %02d, input : %s" % (fn, idx, ymd, hour, str(data)) )

            if not _999_encountered :
                if ( -999 in data ) or ( -999.0 in data ) :
                    _999_encountered = 1
                    print( LINE )
                    log.info( "[%s][%04d] _999_encountered = %s" % (fn, idx, _999_encountered ) )
                    print( LINE )
                pass
            pass

            if len( data_list ) < min_data_list_size :
                data_list.append( data.copy() ) 
            else :
                if _999_encountered : 
                    # x data generation
                    x_data = []

                    for r in data_list :
                        x_data.append( r )
                    pass
                    
                    x_data = np.array(x_data )
                    x_list.append( x_data )

                    # y data generation
                    y_data = np.array( data.copy() )
                    y_list.append( y_data )
                pass

                data_list.pop( 0 )
                data_list.append( data.copy() ) 
            pass                

            ymdh_prev = ymdh
            
            idx += 1                
        pass # // 한 줄 씩 읽기
        
        self.x_list = np.array( x_list )
        self.y_list = np.array( y_list )

        debug and log.info( "[%s] data_distcontinuous_cnt : %d" % ( fn, data_distcontinuous_cnt ) )
        log.info( "[%s] max_y = %s" % (fn, self.max_y ) )
        
    pass # // -- init         

    def __getitem__(self, index):
        return ( self.x_list[index], self.y_list[index] )
    pass

    def __len__(self):
        return len( self.x_list )
    pass

pass

'''
// DataLoader
'''

# TrainCallback
class TrainCallback(callbacks.Callback):
    # TODO: 1006 TrainCallback

    def __init__(self, trafficForeCast ):
        self.loss = -1
        self.trafficForeCast = trafficForeCast
    pass

    def on_train_begin(self, logs=None):
        print("Starting training;" )
    pass

    def on_train_end(self, logs=None):
        print("Stop training;" )
    pass

    def on_epoch_begin(self, epoch, logs=None):
        log.info("\n\nStart epoch %s of training." % ( epoch  + 1 ) )
    pass

    def on_epoch_end(self, epoch, logs=None):
        #  ['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error']
        keys = list(logs.keys())

        #log.info( "%s" % keys )
        
        loss = logs[ "loss" ] #TODO 2010 손실 값 키 설정 loss key set 

        log.info( "\ncurr epoch[%s] val_loss = %s\n" % ( epoch + 1, loss ) )

        if self.loss < 0 or loss < self.loss :
            self.loss = loss

            # 모델 저장 
            trafficForeCast.save_model( "1", self.model )
        pass

        log.info("\nEnd epoch {} of training; got log keys: {}\n".format(epoch, keys)) 
    pass

pass # train callback

#TODO 2001 TrafficForeCast
class TrafficForeCast :
    def __init__(self, config):
        self.config = config
        
        #TODO 1002 dataset path 설정 
        '''
        - 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (tf/notebooks)
        '''
        ds_path = os.path.join('/tf/notebooks/datasets/19_forecast_traffic')

        if os.name is 'nt':  # windows operating system 인 경우
            log.info("os.name : %s" % os.name)        
            ds_path = os.path.join('./datasets')
        pass

        log.info("DATASET_PATH : %s" % ds_path)    

        self.ds_path = ds_path
        # // dataset path
    pass

    #TODO 1008 save model
    def save_model( self, model_name, model ):
        '''
        You can use model.save(filepath) to save a Keras model into a single HDF5 file which will contain:
            the architecture of the model, allowing to re-create the model.
            the weights of the model.
            the training configuration (loss, optimizer)
            the state of the optimizer, allowing to resume training exactly where you left off.
        '''

        root = "./"
        if not model_name.endswith( ".pth") :
            model_name = model_name + '.pth'
        pass

        file_name = os.path.join( root, model_name )
        
        model.save( file_name, overwrite=True, include_optimizer=True, )

        log.info( 'model saved as file_name : %s' % file_name )
    pass # // save model

    #TODO 1009 load model
    def load_model( self, model_name ):
        root = "./"

        if not model_name.endswith( ".pth") :
            model_name = model_name + '.pth'
        pass

        file_name = os.path.join( root, model_name )

        model_loaded = tf.keras.models.load_model( file_name )

        log.info( 'model loaded from %s' % file_name )

        return model_loaded
    pass
    # // load model

    # train model
    def train_model(self) :
        config = self.config 

        num_epochs = config.num_epochs
        model_name = config.model_name
        batch = config.batch 

        log.info( "Model name: %s" % model_name )

        #TODO 1004 데이터 로딩    
        debug = 0 
        ds = {}
        phases = [ "train", "validate" , "test" ]
        ds_path = self.ds_path
        then = time.time()

        for phase in phases : 
            debug = ( phase == "test" )
            debug = 0
            ds[ phase] = TimeSeriesDataset(root=ds_path, phase= phase, debug=debug)
        pass

        now = time.time()
        log.info( "// Training data loading. Duration %d sec(s)" % ( now - then ))
        # // data loading

        x0 = ds["train"].x_list[0]
        y0 = ds["train"].y_list[0]
        
        log.info( "x shape = %s, y shape = %s" % ( x0.shape, y0.shape ) )

        #TODO 1010 모델 구성하기
        log.info( "Model Setting ...." )

        m_name = "_03_flat"

        model = Sequential()

        out_dim = np.prod( x0.shape )
        y_dim = len( y0 )

        log.info( "out_dim = %s, y_dim = %s" % (out_dim, y_dim) )

        model.add( Flatten(input_shape=ds["train"].x_list.shape[1:])) 

        #model.add( LSTM(out_dim, batch_input_shape=(None,out_dim), stateful=True) )

        #TODO 2020 은닉 레이어 숫자
        x_layer_cnt = out_dim
        y_layer_cnt = y_dim

        for _ in range( x_layer_cnt ) :
            model.add( Dense( out_dim ) )
        pass

        for _ in range( y_layer_cnt ):
            model.add( Dense( y_dim ) ) 
        pass

        log.info( "// Done. Model Setting ...." )
        print( LINE )

        #-- 2. 모델 구성.

        # 3. 모델 학습과정 설정하기
        log.info( "Model Compile ...." )
        
        #optimizer = "rmsprop"
        optimizer = "adam"
        loss = "mean_absolute_error" 
        #metrics = "mse"
        metrics = tf.keras.metrics.RootMeanSquaredError()
        
        model.compile( optimizer=optimizer, loss=loss, metrics=[ metrics ] )

        log.info( "// Deon. Model Compile ...." )
        print( LINE )
        #-- 모델 학습 과정 설정.

        # 4. 모델 학습시키기
        print( LINE )
        log.info( "Model Learinig ...." )
        
        # 랜덤 시드 고정 
        np.random.seed(5) 

        # 조기종료 콜백함수 정의
        early_stopping = callbacks.EarlyStopping(patience = 20)

        ds_train = ds[ "train" ]
        ds_validate = ds[ "validate" ]

        hist = model.fit( ds_train.x_list, ds_train.y_list, epochs=num_epochs, batch_size=batch, 
            validation_data=( ds_validate.x_list, ds_validate.y_list ) , 
            callbacks=[ early_stopping, TrainCallback( self ) ] 
        )

        log.info( "// Dene. Model Learinig ...." ) 
        print( LINE )

        # 5. 모델 평가하기

        print( LINE )
        log.info( "Evaluating ....." )

        verbose = 1
        for phase in [ "train" , "validate" ] :
            loss = model.evaluate( ds[phase].x_list, ds[phase].y_list, verbose=verbose ) 
            log.info( 'Loss[%s] : %s' % ( phase, loss ) )
        pass
        
        log.info( "// Evaluating ....." )

        # 6. 결과 보기 
        self.test_model( model, ds["test"], debug = 0 )
        #-- 결과 보기 

        #TODO 2007 학습과정 살펴보기         
        history_loss = hist.history[ 'loss' ]

        max_hist_loss = max( history_loss ) 
        plt.plot( history_loss )
        plt.ylim( 0, max_hist_loss )

        if 0 and max_hist_loss > 100_000 :
            #plt.yscale('log')
            plt.semilogy()
            pass
        pass
        
        plt.ylabel( 'loss' )
        plt.xlabel( 'epoch' )
        plt.legend( ['train'], loc='upper left' ) 
        #plt.show()        
        log.info( "min loss = %s, max_y = %s" % ( (int)(min( history_loss )), (int)(ds["train"].max_y )) )

        # save plot
        plt.savefig( "train_%s.png" % m_name, format="png" )
        #plt.savefig( "train_%s.svg" % m_name, format="svg" )
        
        plt.show()
        #-- 학습 과정 살펴 보기
        
    pass # train model

    # test model
    def test_model(self, model, ds_test ,debug =0 ):
        print( LINE )
        log.info( " Predicting ... ".center( 100, "*") )

        ds = ds_test 
        x_list = ds.x_list
        y_list = ds.y_list

        x_dim = x_list[0].shape[ 1 ] 
        y_dim = y_list.shape[ 1 ]
        debug and log.info( "x dim = %s, y dim = %s" % ( x_dim, y_dim ) )
        #dim = y_list

        #TODO 2021 예상값 생성 generate predictions
        predictions = []
        yList = []

        min_data_list_size = ds_test.min_data_list_size 

        for i in range( len( x_list ) ) :
            x_input = x_list[ i : i + 1 ]            
            
            #TODO 2022 prediction input data reset
            debug and log.info( "x_input_org[%d] = %s" % (i, x_input) )

            for r, y in enumerate( yList ) : 
                for c, yv in enumerate( y ) : 
                    x0 = x_input[0]
                    ridx = min_data_list_size - len( yList ) + r 
                    x = x0[ridx]
                    xv = x[c]
                    if xv == -999.0 or xv == -999 :
                        x[c] = yv

                        debug and log.info( "x[%d, %d] = %s -> %s" % ( ridx, c, xv, yv ) )
                    elif r == len( yList ) - 1 :
                        y[c] = xv
                        debug and log.info( "y[%d] = %s -> %s" % ( c, y[c], xv ) )
                    pass 
                pass                
            pass

            debug and log.info( "x_input_new[%d] = %s" % (i, x_input) ) 
            # -- prediction input data reset 

            result = model.predict( x_input )  #result = model.predict( ds_test.x_list[0:1] )            
            debug and log.info( "result[%d] = %s" % ( i, result ) )

            y = []

            for yv in result[0] :
                if yv < 0 :
                    yv = 0.0
                pass

                #yv = round( yv )
                y.append( yv )
            pass

            debug and log.info( "y_org[%d] = %s" % ( i, y ) ) 

            if len( yList ) == min_data_list_size :
                yList.pop( 0 )
            pass
        
            yList.append( y )            
            predictions.append( y)            
        pass
        #-- generation predictions

        self.save_prediction_file( predictions )

        print( LINE )
        print( " Dene. Predicting. ".center( 100, "*") )
        print( LINE )
    pass # // test model

    #TODO 2002 save prediction file
    def save_prediction_file( self, predictions ): 

        config = self.config 
        output_file = config.prediction_file

        log.info('Write predictions ....')

        if 1 : # txt save
            output = open(output_file, 'w', encoding='utf-8-sig', newline='')

            delim = " "

            for pred in predictions :
                output.write( delim.join([str((int)(x + 0.5)) for x in pred]) + '\n')
            pass

            output.close()

            if os.stat(output_file).st_size == 0:
                raise AssertionError('output result of inference is nothing')
            pass
        pass # // txt save

        if 1 : # csv save
            output = open( "predicition.csv", 'w', encoding='utf-8-sig', newline='')

            delim = ", "
            for pred in predictions :
                output.write( delim.join([str((int)(x + 0.5)) for x in pred]) + '\n')
            pass

            output.close()

            if os.stat(output_file).st_size == 0:
                raise AssertionError('output result of inference is nothing')
            pass
        pass # // csv save

        log.info('Done. Writing predictions.')
    pass #-- save prediction file

pass

pass
#-- Traffic ForeCast

if __name__ == '__main__':
    print( LINE )
    print( " Hello ... ".center( 100, "*") )
    print( LINE )

    # TODO: 1003 시작 함수 main

    # train
    # python main.py 
    # test 
    # python main.py --batch=4 --model_name="1.pth" --mode="test"

    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)  #TODO 2011 훈련수 num ephocs set
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_file", type=str, default="prediction.txt")
    args.add_argument("--batch", type=int, default=24)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    mode = config.mode 

    #mode = "test"  #TODO 2020 모드 강제 설정 set mode forced

    trafficForeCast = TrafficForeCast( config ) 

    if "test" == mode :
        debug = 0
        phase = mode
        ds = {}
        root = trafficForeCast.ds_path
        ds[ phase] = TimeSeriesDataset(root=root, phase=phase, debug=debug )

        model = trafficForeCast.load_model( "1" )
        
        trafficForeCast.test_model( model, ds[ phase], debug =1 )
    else : 
        trafficForeCast.train_model()
    pass

    print( LINE )
    print( ( " mode = %s " % mode ).center( 100, "*" ) ) 
    print( " Good bye! ".center( 100, "*") )
    print( LINE ) 

pass

# end