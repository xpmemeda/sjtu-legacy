import numpy as np 
import matplotlib.pyplot as plt 


np.random.seed(100)
'''
np.random.normal()
	size: output shape
	loc: mean
	scale: standard deviation
'''
data = np.random.normal(size=(1000, 4), loc=0, scale=1)
labels = ['A', 'B', 'C', 'D']


plt.boxplot(data, labels=labels)
plt.show()