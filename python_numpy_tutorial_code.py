# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys

#print sys.getdefaultencoding()

"""
d = {(x, x+1): x for x in range(10)}

t = (5,6)

print(type(t))

print(d[t])

print(d[(1,2)])

x = 3
print(type(x))

print(x+1)

print(x**2)

y = 2.5

print(type(y))

print(y**2)

t = True
f = False
print(type(t))

print(t and f)
print(t or f)
print(not t)
print(t != f)
hello = 'hello'
world = 'world'
hw12 = '%s %s %d' % (hello, world, 12) #sprintf style string formatting
print(type(hw12))

xs = [3, 1, 2]
print(xs, xs[2])

print(type(xs))

xs.append('bar')

print(xs)

x = xs.pop()
print(x)
print(xs)

"""
"""

nums = list(range(5))
print(nums)
print(type(nums))
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8,9]
print(nums)

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
    
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
          
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x**2)
print(squares)

squares = [x**2 for x in nums]
print(squares)

even_squares = [x**2 for x in nums if x%2 == 0]
print(even_squares)

"""

"""

animals = {'cat', 'dog'}
print(animals)
print(type(animals))
print('cat' in animals)
print('fish' in animals)

animals.add('fish')
print(animals)
animals.add('cat')
print(animals)

for idx, animal in enumerate(animals):
    print('#%d:%s' % (idx + 1, animal))
          
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)

"""

import numpy as np

a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[1])
a[0] = 5
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0, 0])
print(b[1,0])
print(b[1,2])

b = np.ones((1,2))
print(b)
print(b.shape)

c = np.full((2,2), 7)
print(c)

d = np.eye(2)
print(d.shape)
print(d)

a = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
b =a[:2, 1:3]
print(a)
print(b)
print(a[0,1])
b[0,0] = 77
print(b)
print(a)

row_r1 = a[1,:] # rank 1 view of the second row of a 
row_r2 = a[1:2, :] #rank 2 view of the second row of a

print(row_r1)
print(row_r1.shape)
print(row_r2)
print(row_r2.shape)

col_r1 = a[:,1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

"如下为使用整数数组索引"
a = np.array([[1,2], [3,4], [5,6]])
print(a)
print(a.shape)
a1 = a[[0, 1, 2], [0, 1, 0]]
print(a1)
print(a1.shape)

a2 = a[[0,0],[1,1]]
print(a2)
print(a2.shape)

a = np.array([[1, 2, 3], [4, 5, 6], [7,8,9], [10, 11, 12]])
print(a)
print(a.shape)

b = np.array([0, 2, 0, 1])
c = np.arange(4)
d = a[c, b]
print(c)
print(d)

a[c, b] += 10
print(a)

a = np.array([[1,2], [3,4], [5,6]])
bool_idx = (a > 2)
print(bool_idx)
b = a[bool_idx]
print(b)
print(b.shape)

print(a[a>2])


x = np.array([1,2])
print(x.dtype)

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1,2], dtype=np.int64)
print(x.dtype)

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x+y)
print(np.add(x,y))

print(y-x)
print(np.subtract(y, x))

print(x*y)
print(np.multiply(x, y))

print(x/y)
print(np.divide(x, y))

print(np.sqrt(x))

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))

print(x.dot(v))
print(np.dot(x, v))
print(x)

print(x.dot(y))
print(np.dot(x, y))

print(np.sum(x))

print(np.sum(x, axis=0))
print(np.sum(x,axis=1))

print(x.T)

v = np.array([1,2,3])
print(v)
print(v.T)

x = np.array([[1,2,3], [4,5,6], [7,8,9],[10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
print(y)

for i in range(4):
    y[i,:] = x[i,:] + v
    
print(y)

vv = np.tile(v, (4,1))
print(vv)

y1 = x + vv
print(y1)

y2 = x + v

print(y2)

v = np.array([1,2,3])
w = np.array([4,5])

print(np.reshape(v, (3,1)) * w)

x = np.array([[1,2,3], [4,5,6]])
print(x+v)

print((x.T + w).T)

print(x + np.reshape(w, (2,1)))
print(w.shape)

w1 = np.reshape(w, (2,1))
print(w1.shape)

print(x*2)