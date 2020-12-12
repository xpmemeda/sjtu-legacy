import numpy as np 
import matplotlib.pyplot as plt 


sales_BJ = [53, 55, 63, 53]
sales_SH = [44, 66, 55, 41]

index = np.arange(4)
print(index)
bar_width = 0.2


'''
left: x-axis coordinate of the columns
bottom: y-axis coordinate of the bottoms of the columns
bar_width: the width of a bar
color: color
'''

plt.bar(left=index+0.1, height=sales_BJ, width=bar_width, color="b")
plt.bar(left=index+0.3, height=sales_SH, width=bar_width, color="r")
plt.bar(left=index+0.1, height=sales_SH, width=bar_width, color="r", bottom=sales_BJ)

plt.show()