# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import logging as log
import logging
log.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

print( "Importing done." )

""" 데이터 다운로드 및 로드 
TensorFlow에서 제공하는 MNIST 데이터 파일 4개를 다운로드하여 data 폴더에 저장하고 읽어옵니다.
최초 실행 시에만 데이터를 다운로드하고, 두 번째 이후부터는 저장된 데이터를 읽어 오기만 합니다. """

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# images/labels 데이터 구조 확인
print( "---- dataset overview --- ")
print( 'train dataset:', mnist.train.images.shape, mnist.train.labels.shape )
print( 'test dataset:', mnist.test.images.shape, mnist.test.labels.shape )
print( 'validation dataset:', mnist.validation.images.shape, mnist.validation.labels.shape )

# 샘플 이미지 출력 

for i in range( 0 ) : 
    im = np.reshape(mnist.train.images[i], [28,28])
    label = np.argmax(mnist.train.labels[i])

    print( '\nlabel :', label )

    plt.imshow(im, cmap='Greys')
    plt.title('label:' + str(label))
    plt.show()
pass

row = 4
col = 10
figsize=(col,row)
fig = plt.figure(figsize=figsize)
img_cnt = 0
for r in range( row ) :
    for c in range( col ) : 
        i = r*col + c
        im = np.reshape( mnist.train.images[i], [28,28])
        label = np.argmax(mnist.train.labels[i])

        ax = fig.add_subplot( row, col, i + 1)
        ax.imshow(im, cmap='Greys')
        ax.text( 2, 2, '%s' % label , color='red', fontweight='bold', horizontalalignment='center' )
    pass
pass

# 회귀 모델 정의 

""" placeholder 정의 : 데이터가 들어 갈 곳 이미지와 정답 레이블용 2차원 tensor를 만든다.
None은 어떤 length도 가능함을 의미한다. """

print( "\n #### Model Definition ##### " )
# 이미지 데이터용 placeholder
x = tf.placeholder(tf.float32, [None, 784])
# 정답 레이블용 placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

""" Variable 정의 : 학습 결과가 저장될 가중치(weight)와 바이어스(bias) """
# 0으로 초기화 함
# w는 784차원의 이미지 벡터를 곱해, 10차원(one hot encoding된 0~9)의 결과를 내기 위한 것
W = tf.Variable(tf.zeros([784, 10])) 
# b는 결과에 더해야 하므로 10차원
b = tf.Variable(tf.zeros([10]))

""" 모델 정의 : Softmax Regression 
10개의 값 중 가장 확률이 높은 것을 고르기 위해 Softmax 사용 """

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 모델 훈련

print( "\n --- Traning now ...." )
import time
then = time.time()

# Loss 함수 정의
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# learning rate을 0.5로 정의
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 세션 시작 전에 모든 변수를 초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 100개씩 샘플링하여 1000회 학습 진행 
for i in range( 1_000 ):
    # 학습 데이터셋에서 무작위로 샘플링한 100개의 데이터로 구성된 'batch'를 가져옴
    batch_xs, batch_ys = mnist.train.next_batch( 10_000 )
    # placeholder x, y_에 샘플링된 batch_xs, batch_ys를 공급함
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
pass

now = time.time()
duration = now - then

print( "\n --- Traning is finished. dration = %f sec" % duration )

# """ 모델 평가 """

print( "\n --- Evaluating model ....." )
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 정확도
print( "\n *** Accuracy : " , sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) )
# // """ 모델 평가 """

# 분류 결과 확인
correct_vals = sess.run(correct_prediction,
                        feed_dict={x: mnist.test.images, y_: mnist.test.labels})
pred_vals = sess.run(y, feed_dict={x: mnist.test.images} )

print( 'All test data ', len(correct_vals), ', correct: ', len(correct_vals[correct_vals == True]), \
      ', wrong:', len(correct_vals[correct_vals == False]) )


# 정확히 분류된 이미지 3개만 확인
fig = plt.figure(figsize=(10,3))
img_cnt = 0
for i, cv in enumerate(correct_vals):
    if cv :  # 정상 분류
        img_cnt +=1
        ax = fig.add_subplot(1,3,img_cnt)
        im = np.reshape(mnist.test.images[i], [28,28])
        label = np.argmax(mnist.test.labels[i])
        pred_label = np.argmax(pred_vals[i])
        ax.imshow(im, cmap='Greys')
        ax.text(2, 2, 'true label=' + str(label) + ', pred label=' + str(pred_label))
    pass

    if img_cnt == 3:  # 3개만 확인
        break
    pass
pass

plt.show()

# end