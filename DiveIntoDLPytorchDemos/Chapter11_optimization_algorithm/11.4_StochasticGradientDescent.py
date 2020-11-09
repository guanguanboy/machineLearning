from d2l import torch as d2l
import math
import torch

f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2 # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2) # gradient

def sgd(x1, x2, s1, s2):
    global lr # learning rate scheduler
    (g1, g2) = gradf(x1, x2)

    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,))
    g2 += torch.normal(0.0, 1, (1,))

    eta_t = eta*lr()# learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0) # Update variables

eta = 0.1
lr = (lambda: 1)
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000))
d2l.plt.show()

def exponential():
    global ctr
    ctr += 1
    return math.exp(-0.1 * ctr)

ctr = 1

lr = exponential

d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000))
d2l.plt.show()

def polynomial():
    global ctr
    ctr += 1
    return (1 + 0.1 * ctr)**(-0.5)

ctr = 1
lr = polynomial
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
d2l.plt.show()