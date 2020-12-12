import math


data = '1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459'
a = data.split(',')
dataset = [(float(a[i]), float(a[i+1])) for i in range(1, len(a)-1, 3)]
# print(dataset)

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
def AGNES(dataset, dist, k):
	C = []; M=[]
	for i in dataset:
		Ci = []
		Ci.append(i)
		C.append(Ci)
	# print(C)
	for i in C:
		Mi = []
		for j in C:
			Mi.append(dist(i,j))
		M.append(Mi)
	# print(M)
	q = len(dataset)
	while q>k:
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
		q -= 1
	return C


C = AGNES(dataset, dist_avg, 3) # the second parameter is a function, so 'dist' in AGNES means dist_avg!