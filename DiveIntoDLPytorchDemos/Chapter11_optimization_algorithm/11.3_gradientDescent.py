from d2l import torch as d2l
import numpy as np
import torch

f = lambda x: x**2 # Objective function  #Lambda表达式就是：能嵌入到其他表达式当中的匿名函数,其中x类似与函数的参数，x**2类似于函数的实现
gradf = lambda x: 2*x #Its derivative

"""
Next, we use  x=10  as the initial value and assume  η=0.2 . Using gradient descent to iterate  x  for 10 times we can see that, eventually, the value of  x  approaches the optimal solution.

"""
def gd(eta):
    x = 10.0
    results = [x]

    for i in range(10):
        x -= eta * gradf(x)
        results.append(float(x))

    print('epoch 10, x:', x)
    return results

res = gd(0.2)

# The progress of optimizing over  x  can be plotted as follows.
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, res], [[f(x) for x in f_line], [f(x) for x in res]],
             'x', 'f(x)', fmts=['-', '-o'])
    d2l.plt.show()

#show_trace(res)

#show_trace(gd(0.05))

#show_trace(gd(1.1))

#local minima
c = torch.tensor(0.15 * np.pi)
f = lambda x: x * torch.cos(c * x)
gradf = lambda x: torch.cos(c * x) - c * x * torch.sin(c * x)

show_trace(gd(2))

def train_2d(trainer, steps=20):
    """Optimize a 2-dim objective function with a customized trainer."""
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))

    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                         torch.arange(-3.0, 1.0, 0.1))

    #print(x1.shape)
    #print(x2.shape)
    #print(x1)
    #print(x2)
    c = d2l.plt.contour(x1, x2, f(x1, x2), 6, colors='#1f77b4') # 6 用来确定轮廓线/区域的数量和位置。
    print(c.levels)  #打印等高线的值
    d2l.plt.clabel(c, inline=True, fontsize=10)
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()


f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2 # Objective

gradf = lambda x1, x2: (2 * x1, 4 * x2)

def gd(x1, x2, s1, s2):
    (g1, g2) = gradf(x1, x2) # Compute gradient
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f, train_2d(gd))

#Newton’s Method
c = torch.tensor(0.5)
f = lambda x: torch.cosh(c * x) # Objective
gradf = lambda x: c * torch.sinh(c * x) #Derivative
hessf = lambda x: c**2 * torch.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * gradf(x) / hessf(x)
        results.append(float(x))

    print('epoch 10, x:', x)
    return results

show_trace(newton())

c = torch.tensor(0.15 * np.pi)
f = lambda x: x * torch.cos(c * x)
gradf = lambda x: torch.cos(c * x) - c * x * torch.sin(c * x)
hessf = lambda x: - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

show_trace(newton())

show_trace(newton(0.5))