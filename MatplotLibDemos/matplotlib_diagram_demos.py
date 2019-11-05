# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:32:43 2019

@author: liguanlin
"""
import numpy as np
import matplotlib.pyplot as plt

#饼图
"""
labels = 'Frogs', 'Hogs','Dogs','Logs'
sizes = [15,30,45,10]
explode=(0,0.1,0,0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.show()
"""

#直方图
np.random.seed(0)

mu,sigma = 100, 20

a = np.random.normal(mu, sigma, size=100)

plt.hist(a, 20, normed=1, histtype='stepfilled', facecolor='b', alpha=0.75)
plt.title('Histogram')

plt.show()
