import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image

import sys
sys.path.append("..") #将自己的搜索目录添加到系统的环境变量中，这种方法是运行时修改，脚本运行后就会失效的
import d2lzh_pytorch as d2l
print(torch.__version__)

data_dir = '/data/pikachu'


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)


assert os.path.exists(os.path.join(data_dir, "train"))

#加载数据集
batch_size, edge_size = 32, 256

train_iter, _ = d2l.load_data_pikachu(batch_size, edge_size, data_dir)
batch = iter(train_iter).next()

#显示图片
imgs = batch["image"][0:10].permute(0, 2, 3, 1)
bboxes = batch["label"][1:10, 0, 1:]

axes = d2l.show_images(imgs, 2, 5).flatten()
for ax, bb in zip(axes, bboxes):
    d2l.show_bboxes(ax, [bb*edge_size], colors=['w'])



