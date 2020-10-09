from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch


def f(x): return x * torch.cos(np.pi * x)
def g(x): return f(x) + 0.2 * torch.cos(5 * np.pi * x)

def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))
    #d2l.plt.show()
    #d2l.plt.gca().show()


x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('empirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('expected risk', (1.1, -1.05), (0.95, -0.5))
d2l.plt.show()

x = torch.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimun', (1.1, -0.95), (0.6, 0.8))
d2l.plt.show()

x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], (0, -0.2), (-0.52, -5.0))
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
d2l.plt.show()

x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, 101 ), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride':10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.show()

x = torch.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
d2l.plt.show()