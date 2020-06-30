# -*- coding: utf-8 -*-

print( " Hello..... ".center( 50, "*") )

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
    
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# TODO: 1002 로그 패키지 임포트
logging.basicConfig(format='%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

LINE = "*"*100

log.info( " Import Done.".center(50, "*") )

def delta_time_series(data):
    a = data[1:]
    b = data[:-1]
    c = a - b
    return c
pass

def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None

    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
        pass
    pass

    y = y.reshape((-1,1))
    
    #log.info( "y.shape : %s" % str(y.shape) )

    return y
pass

def binary_accuracy(a, b):
    '''
    Helper function to compute the match score of two 
    binary numpy arrays.
    '''
    assert len(a) == len(b)
    return (a == b).sum() / len(a)
pass

log.info( " Reading dataset. ".center(50, "*") ) 

dataset_DJI = read_csv('./data_time_series/DJI_20150919-20180918.csv', usecols=[4]).values.astype('float32')
dataset_AAPL = read_csv('./data_time_series/AAPL_20150919-20180918.csv', usecols=[4]).values.astype('float32')
dataset_AMAZN = read_csv('./data_time_series/AMZN_20150919-20180918.csv', usecols=[4]).values.astype('float32')

log.info( " Done. Reading dataset. ".center(50, "*") ) 

dataset_delta_DJI = delta_time_series(dataset_DJI)
dataset_delta_APPL = delta_time_series(dataset_AAPL)
dataset_delta_AMAZN = delta_time_series(dataset_AMAZN)

dataset_x = np.concatenate( (dataset_delta_DJI, dataset_delta_APPL, dataset_delta_AMAZN), axis = 1 )
dataset_y = dataset_delta_DJI

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(dataset_x.reshape(-1, 1))
dataset_x = scaler.transform(dataset_x)
dataset_y = scaler.transform(dataset_y)

# split into train and test sets
train_size = int(len(dataset_x) * 0.67)
test_size = len(dataset_x) - train_size

train_x = dataset_x[0:train_size,:] 
train_y = dataset_y[0:train_size,:] 

test_x = dataset_x[train_size:len(dataset_x),:]
test_y = dataset_y[train_size:len(dataset_y),:]

log.info( "train_x shape: %s" % str( train_x.shape ) )
log.info( "train_y shape: %s" % str( train_y.shape ) )

if 0 :
    plt.plot(delta_time_series(dataset_AMAZN))
    plt.ylabel('Price')
    plt.title( "AMAZN dataset")
    plt.xlabel( 'Day' )
    plt.show(block=1)
pass

mode = "test"

look_back = 3

train_data_gen = TimeseriesGenerator(train_x, train_y,
                    length=look_back, sampling_rate=1,stride=1,
                    batch_size=3) 

test_data_gen = TimeseriesGenerator(test_x, test_y,
                    length=look_back, sampling_rate=1,stride=1,
                    batch_size=1)

if mode == "" : 
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, train_x.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    from tensorflow.keras.utils import plot_model
    model_file_name = 'model_time_series_multi.png'
    plot_model(model, to_file=model_file_name, show_shapes=True)

    if 0 :
        from PIL import Image
        image = Image.open( model_file_name )
        image.show()
    pass

    model.fit_generator(train_data_gen, epochs=10)
    history = model.history

    model_save_file_name = 'dji_model.h'
    model.save( model_save_file_name )
    log.info( "model saved file name = %s." % model_save_file_name )

elif mode == "test" : 
    log.info( "Loading model." )
    model = load_model('dji_model.h')
pass

eval = model.evaluate_generator(test_data_gen)
log.info( "eval = %s" % eval )

trainPredict = model.predict_generator(train_data_gen)
log.info( "trainPredict.shape: %s" % str( trainPredict.shape ) )

testPredict = model.predict_generator(test_data_gen)
log.info( "testPredict.shape: %s" % str( testPredict.shape ) )

# invert predictions, scale values back to real index/price range.
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
dataset_y = scaler.inverse_transform(dataset_y)

trainY_org = get_y_from_generator(train_data_gen)
trainY = scaler.inverse_transform(trainY_org)

testY = get_y_from_generator(test_data_gen)
testY = scaler.inverse_transform(testY)

if 0 : 
    fig, ax = plt.subplots()
    ax.plot( trainY_org, label='trainY_org')
    ax.plot( trainY, label='trainY') 
    
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('1')

    plt.show()
pass

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset_y)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# Delta + previous close
trainPredictPlot = trainPredictPlot + dataset_DJI[1:]

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset_y)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset_y), :] = testPredict

# Delta + previous close
testPredictPlot = testPredictPlot + dataset_DJI[1:]

# plot baseline and predictions

if 1 : 
    fig, ax = plt.subplots()
    ax.plot(dataset_y + dataset_DJI[1:], label='REAL')
    ax.plot(trainPredictPlot, label='train predict')
    ax.plot(testPredictPlot, label='test predict')    

    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

    ax.set_title( "DJI train predict/test Predict" )

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('1')

    plt.show()
pass

acc = binary_accuracy(trainY[:,0]>0,  trainPredict[:,0] >0)
log.info( "trainPredict binary accuracy: %s" % acc )
acc = binary_accuracy(testY[:,0]>0,  testPredict[:,0] >0)
log.info( "testPredict binary accuracy: %s" % acc )

print( " Good bye. ".center(50, "*") )