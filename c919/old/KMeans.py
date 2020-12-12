from pandas import read_csv,DataFrame
from numpy import *
import datetime
from sklearn.preprocessing import MinMaxScaler


print('begin time:',datetime.datetime.now())


def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))



def InitCentroids(dataset, k):
	numrows, numcolumns = dataset.shape
	centroids = zeros((k, numcolumns))
	for i in range(k):
		index = int(random.uniform(0,numrows ))
		centroids[i,:] = dataset[index,:]
	return centroids



def kmean(dataset, k):
	numrows = dataset.shape[0]
	numcolumns = dataset.shape[1]
	ClusterAssment = mat(zeros((numrows,2)))

	ClusterChanged = True

	centroids = InitCentroids(dataset, k)

	while ClusterChanged:
		ClusterChanged = False
		for i in range(numrows):
			minDist = 100.0
			minIndex = 0
			for j in range(k):
				distance = euclDistance(dataset[i,:], centroids[j,:])
				if distance<minDist:
					minDist = distance
					minIndex = j
			ClusterAssment[i,1] = minDist
			if ClusterAssment[i,0]!=minIndex:
				ClusterChanged = True
				ClusterAssment[i,0] =  minIndex
		for j in range(k):
			pointsInCluster = dataset[nonzero(ClusterAssment[:, 0].A == j)[0]]  
			centroids[j, :] = mean(pointsInCluster, axis = 0)  
	return centroids, ClusterAssment



data = read_csv("20171017.csv", index_col=0)
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
dataset = mat(data)
centroids,ClusterAssment = kmean(dataset, 2)
d01 = euclDistance(centroids[0,:], centroids[1,:])
d02 = euclDistance(centroids[0,:], centroids[2,:])
#d03 = euclDistance(centroids[0,:], centroids[3,:])
d12 = euclDistance(centroids[1,:], centroids[2,:])
#d13 = euclDistance(centroids[1,:], centroids[3,:])
#d23 = euclDistance(centroids[2,:], centroids[3,:])
centroids = DataFrame(centroids)
ClusterAssment = DataFrame(ClusterAssment)
centroids.to_csv("centroids.csv", header=False)
ClusterAssment.to_csv("ClusterAssment.csv", header=False, index=False)
#print(" ",d01, d02, d03,"\n",d12, d13,"\n",d23)
print(d01,d02,d12)


print('end time:',datetime.datetime.now())