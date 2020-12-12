import pandas as pd 
import numpy as np 
import tensorflow as tf 
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time


# hyperparameters
rnn_unit = 56
input_size = 56
output_size = 56
lr = 0.001
tf.reset_default_graph()
# Define dimensions for weights and biases
weights = {
		  'in':tf.Variable(tf.random_normal([input_size,rnn_unit], seed=int(5))),
		  'out':tf.Variable(tf.random_normal([rnn_unit,output_size], seed=int(5)))
		  }
biases = {
		  'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
		  'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
		 }


# load dataset
path = 'f:/data/c919/VIB1-56.csv'
dataset = pd.read_csv(path, header=0, index_col=0)
dataset = dataset.values


def get_data(batch_size=60, time_step=20, row_begin=0, row_end=3000):
	batch_index=[]
	data = dataset[row_begin:row_end, :]
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(data)
	x = []
	for i in range(row_end - row_begin - time_step):
		if i%batch_size==0:
			batch_index.append(i)
		x_ = scaled[i:i+time_step, :]
		x.append(x_)
	y = scaled[time_step:, :]
	batch_index.append((row_end-row_begin-time_step))
	return batch_index, x, y, scaler


def lstm(X):
	# # LSTM
	# batch_size = tf.shape(X)[0]
	# time_step = tf.shape(X)[1]
	# cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)
	# #cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
	# init_state = cell.zero_state(batch_size,dtype=tf.float32)
	# output_rnn,final_states=tf.nn.dynamic_rnn(cell, X, initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	# w_out = weights['out']
	# b_out = biases['out']
	# pred = output_rnn[:, -1, :]
	# return pred, w_out, b_out

	# # FC+LSTM
	# batch_size = tf.shape(X)[0]
	# time_step = tf.shape(X)[1]
	# w_in = weights['in']
	# b_in = biases['in']  
	# X = tf.reshape(X,[-1,input_size])  # 将输入转换成2维，便于进行全连接层的矩阵乘法
	# Wx_plus_b_L1 = tf.matmul(X, w_in) + b_in
	# L1 = tf.nn.tanh(Wx_plus_b_L1)  # 激活函数
	# input_rnn = tf.reshape(L1,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
	# cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)
	# #cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
	# init_state = cell.zero_state(batch_size,dtype=tf.float32)
	# output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	# w_out = weights['out']
	# b_out = biases['out']
	# pred = output_rnn[:, -1, :]
	# return pred, w_out, b_out

	# # LSTM+FC
	# batch_size = tf.shape(X)[0]
	# time_step = tf.shape(X)[1]
	# cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)
	# #cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
	# init_state = cell.zero_state(batch_size,dtype=tf.float32)
	# output_rnn,final_states=tf.nn.dynamic_rnn(cell, X,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	# # xxx = tf.subtract(output_rnn[:, -1, :], final_states[1])
	# w_out = weights['out']
	# b_out = biases['out']
	# pred = tf.matmul(output_rnn[:, -1, :], w_out) + b_out
	# pred = tf.nn.tanh(pred)
	# return pred, w_out, b_out

	# FC+LSTM+FC
	batch_size = tf.shape(X)[0]
	time_step = tf.shape(X)[1]
	w_in = weights['in']
	b_in = biases['in']  
	X = tf.reshape(X,[-1,input_size])  # 将输入转换成2维，便于进行全连接层的矩阵乘法
	Wx_plus_b_L1 = tf.matmul(X, w_in) + b_in
	L1 = tf.nn.tanh(Wx_plus_b_L1)  # 激活函数
	input_rnn = tf.reshape(L1,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
	cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)
	#cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
	init_state = cell.zero_state(batch_size,dtype=tf.float32)
	output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	# xxx = tf.subtract(output_rnn[:, -1, :], final_states[1])
	w_out = weights['out']
	b_out = biases['out']
	pred = tf.matmul(output_rnn[:, -1, :], w_out) + b_out
	pred = tf.nn.tanh(pred)
	return pred, w_out, b_out

def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=487):
	X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
	Y = tf.placeholder(tf.float32, shape=[None,output_size])
	# keep_prob = tf.placeholder(tf.float32)
	batch_index_train, train_x, train_y, scaler_train = get_data(batch_size,time_step,train_begin,train_end)
	batch_index_test, test_x, test_y, scaler_test = get_data(batch_size, time_step, 60000, 90000)
	pred , w_out, b_out = lstm(X)
	#define loss function
	loss = tf.reduce_mean(tf.square(pred - Y))
	# loss_ = tf.reduce_mean(tf.square(pred - Y), axis=0)
	train_op = tf.train.AdamOptimizer(lr).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#epoch
		iter_time = 400
		for i in range(iter_time):
			for step in range(len(batch_index_train)-1):
				_ = sess.run(train_op, feed_dict = 
					{X:train_x[batch_index_train[step]:batch_index_train[step+1]], Y:train_y[batch_index_train[step]:batch_index_train[step+1]]})
			# print('iter:', i, time.time)
			if i%10==0:
				loss_train = []
				loss_test = []
				for step in range(len(batch_index_train)-1):
					loss_train_ = sess.run(loss, feed_dict=
						{X:train_x[batch_index_train[step]:batch_index_train[step+1]], Y:train_y[batch_index_train[step]:batch_index_train[step+1]]})
					loss_train.append(loss_train_)
				for step in range(len(batch_index_test)-1):
					loss_test_ = sess.run(loss, feed_dict=
						{X:test_x[batch_index_test[step]:batch_index_test[step+1]], Y:test_y[batch_index_test[step]:batch_index_test[step+1]]})
					loss_test.append(loss_test_)
				loss_train = sum(loss_train) / len(loss_train)
				loss_test = sum(loss_test) / len(loss_test)
				print('iter:',i,'loss_train:',loss_train, 'loss_test:', loss_test, "time:", time.time())
		results = sess.run(pred, feed_dict={X:test_x})
		results = scaler_test.inverse_transform(results)
		labels = scaler_test.inverse_transform(test_y)
		rows, cols = results.shape
		rmse = []
		for i in range(cols):
			rmse_ = sqrt(mean_squared_error(results[:,i], labels[:,i]))
			# rmse_ = rmse_ / (sum(labels) / len(labels))
			# rmse_ = rmse_ / (sum(abs(labels - results)) / len(labels))
			rmse.append(rmse_)
		print("rmse:", rmse)
		# for i in range(rows):
		# 	for j in range(cols):
		# 		if abs(results[i,j]-labels[i,j])/labels[i,j]>0.2:
		# 			print([i,j], results[i,j], labels[i,j])
		# w, b = sess.run([w_out, b_out])
		# print(w, b)
test_predict = train_lstm(batch_size=1000,time_step=15,train_begin=0,train_end=60000)