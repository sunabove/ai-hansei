# -*- coding: utf-8 -*-

# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda  
import tensorflow as tf 
from tensorflow.python.keras import backend as K
import random

# 1. 데이터셋 생성하기
# 입력 x에 대해 2를 곱해 두 배 정도 값을 갖는 출력 y가 되도록 데이터셋을 생성해봤습니다.
# 선형회귀 모델을 사용한다면 Y = w * X + b 일 때, w가 2에 가깝고,
# b가 0.16에 가깝게 되도록 학습시키는 것이 목표입니다.

real = tf.constant( [ 0., 1., 2., 3., 4., 5., 6. ]*100 )
imag = tf.constant( [ 1., 2., 3., 4., 5., 6., 0. ]*100 )
x_train = tf.complex(real, imag) 

real = tf.constant( [ 0., 1., 2., 3., 4., 5., 6. ]*100 )
imag = tf.constant( [ 1., 2., 3., 4., 5., 6., 0. ]*100 )
y_train = tf.complex(real, imag)

real = tf.constant( [ 0., 1., 2., 3., 4., 5., 6. ]*100 )
imag = tf.constant( [ 1., 2., 3., 4., 5., 6., 0. ]*100 )
x_test = tf.complex(real, imag)

real = tf.constant( [ 0., 1., 2., 3., 4., 5., 6. ]*100 )
imag = tf.constant( [ 1., 2., 3., 4., 5., 6., 0. ]*100 )
y_test = tf.complex(real, imag)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=1, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(1)) 

# 3. 모델 학습과정 설정하기
model.compile( optimizer='rmsprop', loss='mse' )

# 4. 모델 학습시키기
from keras.callbacks import EarlyStopping
# 랜덤 시드 고정 
np.random.seed(5)
early_stopping = EarlyStopping(patience = 20) # 조기종료 콜백함수 정의

hist = model.fit(x_train, y_train,
    epochs=50, batch_size=100, steps_per_epoch = 10,
    callbacks=[early_stopping] 
    )

# 5. 학습과정 살펴보기
# %matplotlib inline
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.ylim(0.0, 1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 6. 모델 평가하기
loss = model.evaluate(x_test, y_test, batch_size=32, steps = 10 )

print( '\nLoss : ' + str(loss))


# 7. 주어진 데이터로 추론 모드에서 마지막 층의 출력을 예측하여 넘파이 배열로 반환합니다:
print( "\n--- Result " )

real = tf.constant( [ 6. ] )
imag = tf.constant( [ 0. ] )
x_evaluate = tf.complex(real, imag)

result = model.predict( x_evaluate, steps = 1 )

print( result )