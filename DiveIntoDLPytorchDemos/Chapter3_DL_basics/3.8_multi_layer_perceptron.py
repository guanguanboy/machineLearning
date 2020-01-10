import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

#绘制激活函数
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()

#绘制relu函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
#xyplot(x, y, 'relu')

#绘制relu函数的导数
y.sum().backward()
#xyplot(x, x.grad, 'grad of relu')

y = x.sigmoid()
#xyplot(x, y, 'sigmoid')

#绘制sigmoid的导数
x.grad.zero_()
y.sum().backward()
#xyplot(x, x.grad, 'grad of sigmoid')  #效果，当输⼊为0 时，sigmoid 函数的导数达到最⼤值0.25；当输⼊
#越偏离0 时，sigmoid 函数的导数越接近0。

y = x.tanh()
#xyplot(x, y, 'tanh')

#绘制tanh的导数
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')

