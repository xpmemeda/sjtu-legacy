import numpy as np 
import matplotlib.pyplot as plt 

N = 1000
x = np.random.randn(N)
y = np.random.randn(N)

'''
color: c
size: s
transparency: alpha
shape: marker
'''

plt.scatter(x, y, s=100, c="r", marker="<", alpha=0.5)
plt.show()