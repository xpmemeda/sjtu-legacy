from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import tensorflow as tf
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start = time.time()
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('dataset.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print("values", values[:2, :])    
#print("values.max", values.max())
#print("values.min", values.min())
#print("scaled", scaled[:2, :])    
## frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#print("reframed", reframed.values[:2, :])    
#print("reframed", reframed.values[19:22, :])    
# drop columns we don't want to predict
reframed.drop(reframed.columns[[0,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]], axis=1, inplace=True)
#print("reframed.drop", reframed.values[:2, :])    
#print("reframed.drop", reframed.values[19:22, :])    
reframed.to_csv("dataset_supervised.csv")


# split into train and test sets
values = reframed.values
n_train_lines = 3000
train = values[:n_train_lines, :]
test = values[n_train_lines:, :]
# split into input and outputs
train_X, train_y = train[:-1, :], train[1:,:]
test_X, test_y = test[ :-1, :], test[1:, :]
print("type, train_X", type(train_X))
print("shape, train_X", train_X.shape)

#extend the train_X from 19-d to 1000-d
train_tmp00 = train[:-1, 0:1]
train_tmp01 = np.hstack((train_X, train_tmp00)) 
train_tmp02 = np.hstack((train_tmp01, train_tmp01, train_tmp01, train_tmp01, train_tmp01))
train_tmp03 = np.hstack((train_tmp02, train_tmp02, train_tmp02, train_tmp02, train_tmp02))
train_tmp04 = np.hstack((train_tmp03, train_tmp03))    


#extend the test_X from 19-d to 1000-d
test_tmp10  = test[:-1, 0:1]
test_tmp11  = np.hstack((test_X, test_tmp10))
test_tmp12  = np.hstack((test_tmp11, test_tmp11, test_tmp11, test_tmp11, test_tmp11))
test_tmp13  = np.hstack((test_tmp12, test_tmp12, test_tmp12, test_tmp12, test_tmp12))
test_tmp14  = np.hstack((test_tmp13, test_tmp13))    


print("shape, train_tmp4", train_tmp04.shape)
print("test, train_tmp4", test_tmp14.shape)
train_X = train_tmp04
test_X  = test_tmp14
np.savetxt("./a6.txt",test_X)
#test_X = series_to_supervised(test_X, 1, 1).values
#scaler.fit_transform(test_y)
print("train_X", train_X[:3,:])
print("train_y", train_y[:3,:])
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print("train_X", train_X[:3,:])
print("train_y", train_y[:3,:])

# design network
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(19))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=30, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False)

end = time.time()
print("function time is : ", end-start)

## plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

a = model.get_config()
b = model.get_weights()
c00 = model.get_layer(index = 0).get_weights()[0]
c01 = model.get_layer(index = 0).get_weights()[1]
c02 = model.get_layer(index = 0).get_weights()[2]
#c1 = model.get_layer(index = 1).get_weights()
#c2 = model.get_layer(index = 2).get_weights()
#c = model.legacy_get_config()
print("c00 shape:", len(c00))
print(c00)  
print("c01 shape:", len(c01))
print(c01)  
print("c02 shape:", len(c02))
print(c02)  

start = time.time()
yhat = model.predict(test_X)
end = time.time()
print("predict time is : ", end-start)
##print("c1 shape:", len(c1))
#print(c1)  
#print("c shape:", len(c))
#print(c)  
#print("b shape:", len(b))
#print(b)  
#print("model config is ",a)
