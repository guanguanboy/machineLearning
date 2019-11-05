import torch
import numpy as np
"""
CPU上的所有张量（CharTensor除外）都支持与Numpy的相互转换。
"""
a = torch.ones(5)
a
print(a)

#将torch的Tensor转化为NumPy数组
b = a.numpy()
print(b)

#改变a及b中的值
a.add_(1)
print(a)
print(b)

#将numpy数组转换为torch张量
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

"""
CUDA上的张量
张量可以使用.to方法移动到任何设备（device）上：
"""

# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
x = torch.ones(5)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype