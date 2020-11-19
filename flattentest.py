import numpy as np

a = np.arange(20)
b = a.reshape(4, 5)
print(b)

c = b.flatten().tolist()
print(type(c))
print(c)