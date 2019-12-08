import torch,os, torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

input = torch.randn(3, 2048, 7, 7)

m = nn.AdaptiveAvgPool2d((1,1))

output = m(input)

print(output.shape) #torch.Size([3, 2048, 1, 1])

