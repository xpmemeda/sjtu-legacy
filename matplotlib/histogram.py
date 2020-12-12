import numpy as np 
import matplotlib.pyplot as plt 


mu = 100
sigma = 20
x = mu + sigma * np.random.randn(2000)

'''
bins: the number of groups
color: color
normed: normalize or not
'''

plt.hist(x, bins=20, color="green", normed=True)
plt.show()