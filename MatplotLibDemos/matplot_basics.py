# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
import matplotlib.pyplot as plt
plt.plot([0, 2, 4, 6, 8], [3,1,4,5,2])
plt.ylabel("grade")
plt.axis([-1, 10, 0, 6])
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
def f(t):
    return np.exp(-t)* np.cos(2*np.pi*t)

a = np.arange(0.0, 5.0, 0.02)

plt.subplot(211)
plt.plot(a,f(a))

plt.subplot(2,1,2)
plt.plot(a, np.cos(2*np.pi), 'r--')
"""

"""
a = np.arange(10)
plt.plot(a,a*1.5, 'go-', a*2.5, 'rx', a, a*3.5,'*', a,a*4.5, 'b-.')
plt.show()
"""

"""
matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 20

a = np.arange(0.0, 5.0, 0.02)
plt.plot(a, np.cos(2*np.pi*a), 'r--')

plt.xlabel('横轴： 时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴： 振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2\pi x)$', fontproperties='SimHei', fontsize=25)
#plt.text(2,1, r'$\mu=100$', fontsize=15)
plt.annotate(r'$mu=100$', xy=(2,1), xytext=(3,1.5), arrowprops=dict(facecolor='black', shrink=0.1, width=2))
#plt.plot(a, np.cos(2*np.pi*a), 'r--')
plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()
"""

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(3,3)

ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:-1])
ax3 = plt.subplot(gs[1:,-1])
ax4 = plt.subplot(gs[2,0])
ax4 = plt.subplot(gs[2,1])