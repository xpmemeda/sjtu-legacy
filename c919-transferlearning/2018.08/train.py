from numpy import concatenate
from pandas import DataFrame
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras import layers
import datetime
#print(layers.__file__)

data_lines = 100000
timesteps = 10
train_lines = 100000


time1 = datetime.datetime.now()
print("%s:preprocess data ..."%time1)


#load dataset
dataset = read_csv('data/20170928_51-53.csv', header=0, index_col=0)
values = dataset.values[:(data_lines+timesteps), :]
values = values.astype('float32')
#normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)



train_data_Y = scaled[timesteps:, 0]
train_data_X = scaled[:timesteps, :]
for i in range(timesteps+1,len(scaled)):
	train_data_X = concatenate((train_data_X, scaled[i-timesteps:i, :]), axis=0)
train_data_X = train_data_X.reshape(((int(int(train_data_X.shape[0]) / timesteps)), timesteps, train_data_X.shape[1]))


time2 = datetime.datetime.now()
print("%s:train data ..."%time2)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_data_X.shape[1], train_data_X.shape[2])))
model.add(Dense(1))
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mae', optimizer='Adam')
model.fit(train_data_X[:train_lines, :, :], train_data_Y[:train_lines], epochs=50, batch_size=72, verbose=2, shuffle=False)
model.save_weights('keras/model_weights.h5')


time3 = datetime.datetime.now()
print("preprocess data cost %s\ntrain data cost %s"%(time2-time1, time3-time2))
