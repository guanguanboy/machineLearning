import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

#3 test yolo pretrained model on images
sess = K.get_session()
#sess = tf.compat.v1.keras.backend.get_session()
#3.1 defining classes, anchors and iamge shape
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

#loading a pretrained model
yolo_model = load_model("model_data/yolo.h5")

yolo_model.summary()

#yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
#filtering boxes
#scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#run the graph on a image