from PIL import Image
import numpy as np
import math
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)

#生成多个锚框
d2l.set_figsize()
img = Image.open('img/catdog.jpg')
w, h = img.size
print("w = %d, h = %d" % (w, h))

#MultiBoxPrior方法的功能为指定输入、一组大小和一组宽高比，该函数将返回输入的所有锚框
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_
    computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = [] #pair of （size, sqrt(ration)）
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])

    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])

    pairs = np.array(pairs)

    ss1 = pairs[:, 0] * pairs[:, 1]
    ss2 = pairs[:, 0] / pairs[:, 1]

    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2

    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)

X = torch.Tensor(1, 3, h, w) #构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

boxes = Y.reshape((h, w, 5, 4))
print(boxes[250, 250, 0, :] )#第一个size和ratio分别为0.75和1， 则宽高均为0.75 = 0.7184 + 0.0316 = 0.8206 - 0.0706

"""
d2l.set_figsize()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
d2l.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
d2l.plt.show()
"""

#交并比
"""
通常用来衡量两个边界框的相似度
"""
def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    :param set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :param set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :return:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2,
        shape:(n1, n2)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)) #(n1, n2, 2)
    super_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(super_bounds - lower_bounds, min=0) #(n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] #(n1, n2)

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)

    :param set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :param set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :return:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape:(n1, n2)
    """
    #Find intersections
    intersection = compute_intersection(set_1, set_2) #(n1, n2)

    #Find ares of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) #(n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]) #(n2)

    #Find the union
    #PyTorch auto-broadcasts singletion dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection #(n1, n2)

    return intersection / union #(n1, n2)

#标注训练集的锚框

"""
在训练集中,我们将每个锚框视为一个训练样本.为了训练目标检测模型,我们需要为
每个锚框标注两类标签:一是锚框所含目标的类别,简称类别;
二是真实边界框相对锚框的偏移量,简称偏移量(offset).
在目标检测时,我们首先生成多个锚框,然后为每个锚框预测类别以及偏移量,接着根据预测的偏移量调整
锚框位置从而得到预测边界框,最后筛选需要输出的预测边界框.

我们直到,在目标检测的训练集中,每个图像已标注了真实边界框的位置以及所含目标的类别.
在生成锚框之后,我们主要依据与锚框相似的真实边界框的位置和类别信息为锚框标注.
那么该如何为锚框分配与其相似的真实边界框呢?
"""
bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])

anchors = torch.tensor([[0, 0.1, 0.2, 0.3],
                        [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

"""
fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()
"""
compute_jaccard(anchors, ground_truth[:, 1:]) #验证一下写的compute_jacard函数

labels = d2l.MultiBoxTarget(anchors.unsqueeze(dim=0),
                       ground_truth.unsqueeze(dim=0))

print(labels[2])

print(labels[1])

print(labels[0])
#输出预测边界框
"""
在模型预测阶段,我们先为图像生成多个锚框,并为这些锚框意义预测类别和偏移量.
随后,我们根据锚框及其预测偏移量得到预测边界框.
当锚框数量较多时,同一个目标上可能会输出较多相似的预测边界框.为了使得结果更加简洁,
我们可以移除相似的预测边界框.常用的方法叫做:非极大值抑制(non-maximun suppression, 简称NMS)
"""

"""
非极大值抑制算法:
对于一个19*19的网格,你会得到一个19*19*8的输出量
在我们的例子中,我们只做car 检测(单目标的检测),因此我们去掉c1, c2, c3这三个类.
这时,每个输出的预测是[pc, bx, by, bh, bw],其中pc是含有car的概率
bx,by, bh,bw是bounding box

step 1: discard all boxes with pc <= 0.6(阈值)

step 2:
    While there are any remaining boxes:
    1, Pick the box with the largest pc, Output that as a prediction.
    2, Discard any remaining box with IoU >= 0.5 with the box output in the previous step
    
    直到所有的网格要么被输出为预测结果,要么被丢弃
    
对于多个目标检测的非极大值抑制算法,就是针对每个目标,分别运行一次step1和step2
"""

"""
下面是一个具体的例子.先构造四个锚框.为了简单起见,我们假设预测偏移量全是0:预测边界框
即锚框.最后,我们构造每个类别的预测概率.
"""
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91],
                        [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,], #背景的预测概率
                          [0.9, 0.8, 0.7, 0.1], #狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]]) #猫的预测概率
"""
fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
d2l.plt.show()
"""

output = d2l.MultiBoxDetection(
    cls_probs.unsqueeze(dim=0),
    offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0),
    nms_threshold=0.5
)
"""
返回结果output形状为(批量大小,锚框个数, 6).
其中每一行6个元素代表同一个预测边界框的输出信息.
第一个元素是索引从0开始计数的预测类别(0为狗, 1为猫),
其中-1表示背景或在非极大值抑制中被移除.第二个元素是预测边界框的置信度.
剩余四个元素分别是预测边界框左上角的x, y 轴坐标和右下角的x, y轴坐标(值域在0和1之间)
"""
print(output)

#移除掉类别为-1的预测边界框,并可视化非极大值抑制保留的结果.
fig = d2l.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ('dog', 'cat')[int(i[0])] + str(i[1])
    d2l.show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

d2l.plt.show()