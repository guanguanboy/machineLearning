import torch
from time import time

print(torch.__version__)

a = torch.ones(1000)
b = torch.ones(1000)

#将这两个向量按元素逐一做标量加法：
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)


#将两个向量直接做矢量加法：
start = time()
d = a + b
print(time() - start)

#广播机制的例子：
a = torch.ones(3)
b = 10
print(a + b)

