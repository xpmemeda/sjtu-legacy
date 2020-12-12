import tensorflow as tf 
import numpy as np 
import pandas as pd 
import datetime

lr = 0.01

batch_size = 80
time_step = 10
size = 3

def get_batch(start, end, batch_size=batch_size):
	batch_index = []
	n_batch = int((end-start)/batch_size) + 1
	for i in range(n_batch+1):
		if i==n_batch:
			batch_index.append(end)
		else:
			batch_index.append(start + batch_size * i)
	return batch_index

def getdata(time_step=time_step):
	dataset = pd.read_csv('data/20171017_51-53.csv', header=0)
	dataset = np.array(dataset)
	dataset = dataset[:10000]
	size = dataset.shape[-1]
	Max = [np.max(dataset[:, i]) for i in range(size)]
	Min = [np.min(dataset[:, i]) for i in range(size)]
	for i in range(size):
		dataset[:, i] = (dataset[:, i] - Min[i]) / (Max[i] - Min[i])
	x = np.array([])
	y = np.array([])
	for i in range(time_step, dataset.shape[0]):
		x = np.append(x, dataset[i-time_step:i])
		y = np.append(y, dataset[i])
	x = np.reshape(x, (-1, time_step, size))
	y = np.reshape(y, (-1, size))
	return x, y, Max, Min 

def train(batch_size=batch_size, time_step=time_step, train_start=0, train_end=7000, test_start=7000, test_end=10000):
	# get data
	x, y, Max, Min = getdata(time_step=time_step)
	# get batch 
	batch_index_train = get_batch(train_start, train_end, batch_size)
	batch_index_test = get_batch(test_start, test_end, batch_size)
	with tf.Session() as sess:
		#load graph and weights
		saver = tf.train.import_meta_graph('rnn/my-model-990.meta') # load graph
		saver.restore(sess, tf.train.latest_checkpoint("rnn/")) # load weights
		graph = tf.get_default_graph()
		x_ = graph.get_tensor_by_name("input:0")
		y_ = graph.get_tensor_by_name('label:0')
		output = graph.get_tensor_by_name('rnn/rnn/transpose_1:0')
		input_fc2 = graph.get_tensor_by_name('fc2/strided_slice:0')
		pred = graph.get_tensor_by_name('fc2/Sigmoid:0')
		loss = graph.get_tensor_by_name('Sqrt:0')
		# train_op = graph.get_operation_by_name('GradientDescent')
		# output = tf.stop_gradient(output)
		# input_fc2 = output[:, -1, :]
		pred = tf.stop_gradient(pred)
		# fine tuning
		weights = tf.Variable(tf.truncated_normal((3, 3), stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=(1, 3)))
		plus = tf.matmul(pred, weights) + biases
		fc = tf.nn.sigmoid(plus)
		# define new loss function and train op.
		loss_ft = tf.sqrt(tf.reduce_mean(tf.square(fc - y_)))
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_ft)
		
		iter_time = 401
		sess.run(tf.global_variables_initializer())
		print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
		loss_train = sess.run(loss, feed_dict={x_:x[train_start:train_end],
											   y_:y[train_start:train_end]})
		loss_test  = sess.run(loss, feed_dict={x_:x[test_start:test_end], 
											   y_:y[test_start:test_end]})
		print('loss_train: ' + str(loss_train) + '   loss_test: ' + str(loss_test))
		for epoch in range(iter_time):
			for step in range(len(batch_index_train)-1):
				# print(step)
				sess.run(train_op, feed_dict={x_:x[batch_index_train[step]:batch_index_train[step+1]], 
											  y_:y[batch_index_train[step]:batch_index_train[step+1]]})
			if epoch%10==0:
				# loss_ = sess.run(loss, feed_dict={x_:x[train_start:train_end],
				# 								  y_:y[train_start:train_end]})
				# print(loss_)
				loss_train = sess.run(loss_ft, feed_dict={x_:x[train_start:train_end],
													   y_:y[train_start:train_end]})
				loss_test  = sess.run(loss_ft, feed_dict={x_:x[test_start:test_end], 
													   y_:y[test_start:test_end]})
				print('loss_train: ' + str(loss_train) + '	loss_test: ' + str(loss_test))
		print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
tf.reset_default_graph()
train(batch_size=batch_size, time_step=time_step, train_start=0, train_end=3000, test_start=3000, test_end=10000)