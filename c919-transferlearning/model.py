import pandas as pd 
import numpy as np 
import tensorflow as tf 
import datetime 

lr = 0.01

batch_size = 80
time_step = 10
size = 3

tf.reset_default_graph()

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
	dataset = pd.read_csv('data/20170928_51-53.csv', header=0)
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
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
def net(x):
	batch_size = tf.shape(x)[0]
	# # Fc1:
	# with tf.variable_scope("fc1"):
	# 	input_fc1 = tf.reshape(x, (-1, time_step*size))
	# 	w_fc1 = tf.Variable(tf.random_normal((time_step*size, 128)), name='w')
	# 	b_fc1 = tf.Variable(tf.constant(0.1, shape=(1, 128)), name='b')
	# 	wx_plus_b_L1 = tf.add(tf.matmul(input_fc1, w_fc1), b_fc1)
	# 	fc1 = tf.nn.tanh(wx_plus_b_L1)
	# # Fc2:
	# with tf.variable_scope("fc2"):
	# 	w_fc2 = tf.Variable(tf.random_normal((128, 3)), name='w')
	# 	b_fc2 = tf.Variable(tf.constant(0.1, shape=(1, 3)), name='b')
	# 	Wx_plus_b_L2 = tf.add(tf.matmul(fc1, w_fc2), b_fc2)
	# 	fc2 = tf.nn.tanh(Wx_plus_b_L2)
	# RNN：
	with tf.variable_scope("rnn"):
		input_rnn = x # shape=(?, 10, 3)
		# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=50)
		# print(cell)
		multi_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(5)])
		initial_state = multi_cell.zero_state(batch_size, np.float32)
		output, final_state = tf.nn.dynamic_rnn(multi_cell, input_rnn, initial_state=initial_state)
	# # lstm:
	# with tf.variable_scope("lstm"):
	# 	input_lstm = x
	# 	cell = tf.contrib.rnn.BasicLSTMCell(50)
	# 	initial_state = cell.zero_state(batch_size, dtype=tf.float32)
	# 	output, final_states = tf.nn.dynamic_rnn(cell, input_lstm, initial_state=initial_state) 
	# 	# xxx = tf.subtract(output_rnn[:, -1, :], final_states[1])
	#fc2:
	with tf.variable_scope("fc2"):
		input_fc2 = output[:, -1, :]
		w_fc2 = tf.Variable(tf.random_normal((128, size)), name="w")
		b_fc2 = tf.Variable(tf.constant(0.1, shape=(1, size)), name="b")
		wx_plus_b_L2 = tf.add(tf.matmul(input_fc2, w_fc2), b_fc2)
		fc2 = tf.nn.sigmoid(wx_plus_b_L2) 
	
	return fc2


def train(batch_size=batch_size, time_step=time_step, train_start=0, train_end=7000, test_start=7000, test_end=10000):
	x, y, Max, Min = getdata(time_step=time_step)

	batch_index_train = get_batch(train_start, train_end, batch_size)
	batch_index_test = get_batch(test_start, test_end, batch_size)

	x_ = tf.placeholder(tf.float32, shape=[None, time_step, size], name="input")
	y_ = tf.placeholder(tf.float32, shape=[None, size], name="label")

	pred = net(x_) 
	loss = tf.sqrt(tf.reduce_mean(tf.square(pred - y_)))
	train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
	# train_op = tf.train.AdamOptimizer(lr).minimize(loss)

	# saver = tf.train.Saver(max_to_keep=4)
	# tf.add_to_collection("predict", pred)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		iter_time = 401
		# start training
		print(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
		loss_train = sess.run(loss, feed_dict={x_:x[train_start:train_end],
											   y_:y[train_start:train_end]})
		loss_test  = sess.run(loss, feed_dict={x_:x[test_start:test_end], 
											   y_:y[test_start:test_end]})
		print('   loss_train: ' + str(loss_train) + '   loss_test: ' + str(loss_test))
		for epoch in range(iter_time):
			for step in range(len(batch_index_train)-1):
				# print(step)
				sess.run(train_op, feed_dict={x_:x[batch_index_train[step]:batch_index_train[step+1]], 
											  y_:y[batch_index_train[step]:batch_index_train[step+1]]})
			if epoch%10==0:
				loss_train = sess.run(loss, feed_dict={x_:x[train_start:train_end],
													   y_:y[train_start:train_end]})
				loss_test  = sess.run(loss, feed_dict={x_:x[test_start:test_end], 
													   y_:y[test_start:test_end]})
				print('iter:' + str(epoch) + '   loss_train: ' + str(loss_train) + '   loss_test: ' + str(loss_test))

				# saver.save(sess, "rnn/my-model", global_step=epoch)
				# print("save the model")
	print(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
train(batch_size=batch_size, time_step=time_step, train_start=0, train_end=3000, test_start=3000, test_end=10000)