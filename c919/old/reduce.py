#截取部分文档
with open('170928.txt','r') as fileread:
	filewrite = open('data.txt','w')
	i = 0
	while i < 10000:
		a = fileread.readline()
		filewrite.write(a)
		i = i + 1
filewrite.close()