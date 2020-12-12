import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(-10, 10, 100) # Average the interval [-10, 10] with 100 spot.
y = x ** 2

'''
color: color
transparency: alpha
shape: marker
'''

plt.plot(x, y, color="r", marker="<", alpha=0.1)
plt.show()