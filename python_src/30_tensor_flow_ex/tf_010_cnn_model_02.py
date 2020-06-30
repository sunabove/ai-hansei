# -*- coding: utf-8 -*-

'''
문제 형태 : 다중 클래스 분류
입력 : 손으로 그린 삼각형, 사각형, 원 이미지
출력 : 삼각형, 사각형, 원일 확률을 나타내는 벡터
'''

import os 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정시키기
np.random.seed(3)

'''
본 예제에서는 패치 이미지 크기를 24 x 24로 하였으니 target_size도 (24, 24)로 셋팅하였습니다.
훈련 데이터 수가 클래스당 15개이니 배치 크기를 3으로 지정하여 총 5번 배치를 수행하면 하나의 epoch가 수행될 수 있도록 하였습니다.
다중 클래스 문제이므로 class_mode는 ‘categorical’로 지정하였습니다.
그리고 제네레이터는 훈련용과 검증용으로 두 개를 만들었습니다.
'''

train_datagen = ImageDataGenerator(rescale=1./255)

path = './data/handwriting_shape'

train_generator = train_datagen.flow_from_directory(
    os.path.join( path, "train" ) ,
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join( path, "test" ),
    target_size=(24, 24),    
    batch_size=3,
    class_mode='categorical')

'''
컨볼루션 레이어 : 입력 이미지 크기 24 x 24, 입력 이미지 채널 3개, 필터 크기 3 x 3, 필터 수 32개, 활성화 함수 ‘relu’
컨볼루션 레이어 : 필터 크기 3 x 3, 필터 수 64개, 활성화 함수 ‘relu’
맥스풀링 레이어 : 풀 크기 2 x 2
플래튼 레이어
댄스 레이어 : 출력 뉴런 수 128개, 활성화 함수 ‘relu’
댄스 레이어 : 출력 뉴런 수 3개, 활성화 함수 ‘softmax’
'''

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

'''
loss : 
    현재 가중치 세트를 평가하는 데 사용한 손실 함수 입니다.
    다중 클래스 문제이므로 ‘categorical_crossentropy’으로 지정합니다.
optimizer :
    최적의 가중치를 검색하는 데 사용되는 최적화 알고리즘으로 효율적인 경사 하강법 알고리즘 중 하나인 ‘adam’을 사용합니다.
metrics :
    평가 척도를 나타내며 분류 문제에서는 일반적으로 ‘accuracy’으로 지정합니다
'''

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print( model.summary() )

# 학습
'''
첫번째 인자 : 훈련데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 train_generator으로 지정합니다.
steps_per_epoch : 한 epoch에 사용한 스텝 수를 지정합니다. 총 45개의 훈련 샘플이 있고 배치사이즈가 3이므로 15 스텝으로 지정합니다.
epochs : 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정합니다. 100번을 반복적으로 학습시켜 보겠습니다.
validation_data : 검증데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 validation_generator으로 지정합니다.
validation_steps : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 홍 15개의 검증 샘플이 있고 배치사이즈가 3이므로 5 스텝으로 지정합니다.
'''

print( "\Learning ...." )

model.fit_generator(
    train_generator,
    steps_per_epoch=15,
    epochs=50,
    validation_data=test_generator,
    validation_steps=5
    )    

# 모델 평가

print("\n-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 모델 사용

print("\n-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

# end