from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch

f = lambda x: 0.5 * x ** 2 # Convex
g = lambda x: torch.cos(np.pi * x ) # Nonconvex
h = lambda x: torch.exp(0.5 * x) # Convex
x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)

d2l.plt.show()

f = lambda x: (x-1)**2 * (x+1)
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
d2l.plt.show()

f = lambda x: 0.5 * x ** 2
x = torch.arange(-2, 2, 0.01)
axb, ab = torch.tensor([-1.5, -0.5, 1]), torch.tensor([-1.5, 1])
d2l.set_figsize()
d2l.plot([x, axb, ab], [f(x) for x in [x, axb, ab]], 'x', 'f(x)')
d2l.annotate('a', (-1.5, f(-1.5)), (-1.5, 1.5))
d2l.annotate('b', (1, f(1)), (1, 1.5))
d2l.annotate('x', (-0.5, f(-0.5)), (-1.5, f(-0.5)))
d2l.plt.show()