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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# TODO: 1002 로그 패키지 임포트
logging.basicConfig(format='%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

LINE = "*"*100

log.info( " Import Done.".center(50, "*") )

dataset_DJI = read_csv('./data_time_series/DJI_20150919-20180918.csv', usecols=[4]).values.astype('float32')
dataset_AAPL = read_csv('./data_time_series/AAPL_20150919-20180918.csv', usecols=[4]).values.astype('float32')
dataset_AMAZN = read_csv('./data_time_series/AMZN_20150919-20180918.csv', usecols=[4]).values.astype('float32')

log.info( " Done. Reading dataset. ".center(50, "*") )

def delta_time_series(data):
    a = data[1:]
    b = data[:-1]
    c = a - b
    return c
pass

def plot_delta(data):
    plt.plot(delta_time_series(data))
    plt.ylabel('close')
    plt.show(block=0)
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
    
    log.info( "y.shape : %s" % str(y.shape) )

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

dataset_delta_DJI = delta_time_series(dataset_DJI)
dataset_delta_APPL = delta_time_series(dataset_AAPL)
dataset_delta_AMAZN = delta_time_series(dataset_AMAZN)

if 0 : 
    plot_delta(dataset_AMAZN)
pass

# Single time series as input
# Normalize datasets

dataset = dataset_delta_DJI
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size

train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

look_back = 3

train_data_gen = TimeseriesGenerator(train, train,
                length=look_back, sampling_rate=1,stride=1,
                batch_size=3)

test_data_gen = TimeseriesGenerator(test, test,
                length=look_back, sampling_rate=1,stride=1,
                batch_size=1)

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

from tensorflow.keras.utils import plot_model
model_file_name = "model_time_series_single.png"
plot_model(model, to_file='model_file_name', show_shapes=True)

if 0 :
    from PIL import Image
    image = Image.open( model_file_name )
    image.show()
pass

model.fit_generator(train_data_gen, epochs=10)
history = model.history

model.evaluate_generator(test_data_gen)

trainPredict = model.predict_generator(train_data_gen)
log.info( "trainPredict shape: %s" % str( trainPredict.shape ) )

testPredict = model.predict_generator(test_data_gen)
log.info( "testPredict shape: %s" % str( testPredict.shape ) )

# invert predictions, scale values back to real index/price range.
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainY = get_y_from_generator(train_data_gen)
testY = get_y_from_generator(test_data_gen)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
log.info('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
log.info('Test Score: %.2f RMSE' % (testScore))

dataset = scaler.inverse_transform(dataset)
log.info( "dataset.shape : %s" % str( dataset.shape ) )

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# Delta + previous close
trainPredictPlot = trainPredictPlot + dataset_DJI[1:]
# set empty values
# trainPredictPlot[0:look_back, :] = np.nan
# trainPredictPlot[len(trainPredict)+look_back:, :] = np.nan

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# Delta + previous close
testPredictPlot = testPredictPlot + dataset_DJI[1:]
# set empty values
# testPredictPlot[0:len(trainPredict)+(look_back*2), :] = np.nan
# testPredictPlot[len(dataset):, :] = np.nan

# plot baseline and predictions
plt.plot(dataset + dataset_DJI[1:])
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show() 

print( " Good bye. ".center(50, "*") )