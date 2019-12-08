from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)))

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)))

    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(0.0, basic_loss))

    return loss

#测试tripelat 函数
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)
    print("loss =" + str(loss.eval()))

#load the trained model
FRmodel.compile(optimizer='adam', loss= triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

#3applying the model
#3.1 face verification
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    encoding = img_to_encoding(image_path, model)

    distance = np.linalg.norm(encoding - database[identity])

    if distance < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return distance, door_open

verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg","kian", database, FRmodel)

#3.2 face recognition
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image_path, FRmodel)

    min_dist = 100

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("not in the database")
    else:
        print("It's " + str(identity) + ", distance is " + str(min_dist))

    return min_dist, identity

who_is_it("images/camera_0.jpg", database, FRmodel) #camera_0.jpg is picture of younes

who_is_it("images/camera_1.jpg", database, FRmodel) #camera_1.jpg is picture of bertrand


"""
Your Happy House is running well. It only lets in authorized persons, and people don't need to carry an ID card around anymore! 

You've now seen how a state-of-the-art face recognition system works.

Although we won't implement it here, here're some ways to further improve the algorithm:
•Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then given a new image, compare the new face to multiple pictures of the person. This would increae accuracy.
•Crop the images to just contain the face, and less of the "border" region around the face. This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.

"""
