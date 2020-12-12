import math
import cv2
import numpy as np 

path = 'F:/Data/ir_images/dataset0/0171.jpg'
img = cv2.imread(path, 0)
x, y = img.shape
# print(x,y) # x = 512, y = 640


def dist(a, b):
	return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
def dist_min(Ci, Cj):
	return min(dist(i, j) for i in Ci for j in Cj)
def dist_max(Ci, Cj):
	return max(dist(i, j) for i in Ci for j in Cj)
def dist_avg(Ci, Cj):
	return sum(dist(i, j) for i in Ci for j in Cj) / (len(Ci) * len(Cj))
def find_Min(M):
	min = 1000
	x = 0; y = 0
	for i in range(len(M)):
		for j in range(len(M)):
			if i!=j and M[i][j]<min:
				min = M[i][j]; x = i; y = j
	return (x, y, min)
# find out the points whose grayscale are more then 200
def getpoint(img):
	l = []
	for i in range(x):
		for j in range(y):
			if img[i, j]>=200:
				l.append([i, j])
	return l 
def getobject(C):
	length = 0
	for i in C:
		if len(i)>length:
			length = len(i)
			object_points = i
	return object_points
def AGNES(dataset, dist):
	C = []; M=[]
	for i in dataset:
		Ci = []
		Ci.append(i)
		C.append(Ci)
	for i in C:
		Mi = []
		for j in C:
			Mi.append(dist(i,j))
		M.append(Mi)
	s = 0
	while s<10:
		x, y, min = find_Min(M)
		C[x].extend(C[y])
		C.remove(C[y])
		# print(C)
		M = []
		for i in C:
			Mi = []
			for j in C:
				Mi.append(dist(i,j))
			M.append(Mi)
		s = min
	return C


l = getpoint(img)
C = AGNES(l, dist_avg) # the second parameter is a function, so 'dist' in AGNES means dist_avg!
object_points = getobject(C)

X, Y = np.mean(np.array(object_points),0)
X = int(X); Y = int(Y)
img[X-10:X+10,Y-10:Y+10]=255
cv2.imwrite('predict.jpg', img)
cv2.imshow('',img)
cv2.waitKey(10000)