import tensorflow as tf
print(tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

from keras import backend as K
print(tf.test.is_gpu_available())