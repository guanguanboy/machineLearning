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