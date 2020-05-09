#多尺度目标检测
"""
    在"锚框"一节中,我们在实验中以输入图像的每个像素为中心生成多个锚框.
这些锚框是对输入图像不同区域的采样.
然而,如果以图像每个像素为中心都生成锚框,很容易生成过多锚框而
造成计算量过大.
减少锚框个数并不难.一个简单的方法是在输入图像中均匀采样一小部分像素,并以采样的像素为
中心生成锚框.
此外,在不同尺度下,我们可以生成不同数量和不同大小的锚框.
值得注意的是,较小目标比较大目标在图像中出现位置的可能性更多.
因此,当使用较小的锚框来检测较小目标时,我们可以采样较多的区域
而当使用较大锚框来检测较大目标时,我们可以采样较少的区域.
"""
from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)
img = Image.open('img/catdog.jpg')
w, h = img.size
print(w, h)

d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    #平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)

display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
d2l.plt.show()