import numpy as np 
import matplotlib.pyplot as plt 


labels = ['A', 'B', 'C', 'D']
fracs = [15, 30, 45, 10]
explode = [0, 0.05, 0, 0]

'''
autopct: display ratio
explode: The distance between the fan and the center of the circle
'''

plt.axes(aspect=1)
plt.pie(x=fracs, labels=labels, autopct='%.0f%%', explode=explode)
plt.show()