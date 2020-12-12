#计算文本行数
import datetime
a = input('The file you want to count:')
print('begin time:',datetime.datetime.now())
with open(a,'r') as f:
	i = 0
	for line in f:
		i = i+1
print(i)
print('end time:',datetime.datetime.now())