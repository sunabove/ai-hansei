from __future__ import absolute_import, division, print_function, unicode_literals

print( "------ Hello -----" )
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys
print( "Python version : %s" % sys.version )

print( "TensorFlow version : %s" % tf.__version__)

print( "Keras version : %s" % keras.__version__ )

import torch
print( "Torch version : %s" % torch.__version__ )
import torchvision
print( "Torchvision version : %s" % torchvision.__version__ )

# OpenCV version
try:
    # % opencv version on only exists in Colab.
    import cv2
    print( "OpenCV version : %s" % cv2.__version__ )
except Exception as e :
    print( e )
    print( "OpenCV is not installed on this machine.")
pass

# print gpu spec
if 0 : 
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
pass

import cv2
if 0 : 
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    pass
pass
# // print gpu spec

print( "------ Good bye! ------" )