from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


data_lines = 100000
timesteps = 10
transfer_lines = 1000


time1 = datetime.datetime.now()
print("%s:preprocess data ..."%time1)


#load dataset
dataset = read_csv('F:\\Data\\keras-transfer-learning-for-airplane\\20170928_51-53.csv', header=0, index_col=0)
values = dataset.values[:(data_lines+timesteps), :]
values = values.astype('float32')
#normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

test_data_Y = scaled[timesteps:, 0]
test_data_X = scaled[:timesteps, :]
for i in range(timesteps+1,len(scaled)):
	test_data_X = concatenate((test_data_X, scaled[i-timesteps:i, :]), axis=0)
test_data_X = test_data_X.reshape(((int(int(test_data_X.shape[0]) / timesteps)), timesteps, test_data_X.shape[1]))


time2 = datetime.datetime.now()
print("%s:transfer and predict data ..."%time2)


model = Sequential()
model.add(LSTM(50, input_shape=(test_data_X.shape[1], test_data_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.load_weights('F:\\Data\\keras-transfer-learning-for-airplane\\model_weights.h5')
model.fit(test_data_X[:transfer_lines,:,:], test_data_Y[:transfer_lines], epochs=3, batch_size=72, verbose=2, shuffle=False)
predict = model.predict(test_data_X)

predict = concatenate((predict,scaled[timesteps:, 1:]), axis=1)
predict = scaler.inverse_transform(predict)
predict = predict[:, 0]

#rmse = sqrt(mean_squared_error(predict, values[timesteps:, 0]))
rmse = sqrt(mean_squared_error(predict[transfer_lines:], values[(timesteps+transfer_lines):, 0]))
print('Test RMSE: %.3f' % rmse)


time3 = datetime.datetime.now()
print("preprocess data cost %s\npredict data cost %s"%(time2-time1, time3-time2))