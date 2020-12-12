import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    #loaddata
	# df = pd.read_csv("./dataset_2.csv")
    #testdata
	# xs = tf.constant([[[1,3,5,5,1]],[[8,9,1,3,5]],[[4,4,1,3,5]],[[1,5,1,3,5]],[[4,5,1,3,5]],[[4,1,5,1,5]],[[4,4,1,5,1]],[[1,5,1,3,5]],[[4,5,1,3,5]],[[1,5,1,3,5]]], tf.float32)
	# x4 = tf.constant([[[2,0,3,5,1]],
	# 			       [[2,0,1,5,1]],
	# 			       [[1000,1000,500,5,1000]],
	# 				   [[4, 4, 1, 5, 1]]], tf.float32)


	learning_rate = 0.01
	time_step = 4
	steps=10

	_X = tf.placeholder(tf.float32, shape=(None,5))

	#add the dnn laywer
	w1 = tf.Variable(tf.random_normal([5,50], stddev=1, seed=1))
	layer1 = tf.matmul(_X, w1)
	w2 = tf.Variable(tf.random_normal([50,100], stddev=1, seed=1))
	layer2 = tf.matmul(layer1, w2)
	w3 = tf.Variable(tf.random_normal([100,120], stddev=1, seed=1))
	layer3 = tf.matmul(layer2, w3)  #layer3.get_shape: (None, 120)
	xa = tf.reshape(layer3, [12, 4, 30])
	#xal = tf.reshape(_X, [12,1,5])


	#LSTM
	cell = tf.nn.rnn_cell.BasicLSTMCell(30, forget_bias=1.0, state_is_tuple=True)
	init_state = cell.zero_state(12, dtype=tf.float32)
	xallhaha, final_state = tf.nn.dynamic_rnn(cell, xa, initial_state=init_state, time_major=False)
	# lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=1.0, state_is_tuple=True)
	# lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=1.0, state_is_tuple=True)
	# all, state1 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, xal, dtype=tf.float32)
	# xall = all[1]
	print(xallhaha.get_shape().as_list())  # [12, 4, 30] 
	xall = tf.reshape(xallhaha, [12, 1, 120])
	print(xall.get_shape().as_list())  # [12, 1, 120]
	x3 = tf.strided_slice(xall, [0], [10], [1])  # strided_slice(input, begin, end, strides=None, ...)
	x4 = tf.strided_slice(xall, [10], [12], [1])

	_Y = tf.placeholder(tf.float32)
	yall = tf.reshape(_Y, [12,1])
	y3 = tf.strided_slice(yall, [0], [10])
	y4 = tf.strided_slice(yall, [10], [12])


	# lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=0, state_is_tuple=False)
	# lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=0, state_is_tuple=False)
	# x_fb, state1 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, xs, dtype=tf.float32)
	# x3 = x_fb[0]
	# print(x3.get_shape().as_list())
	# print(x4.get_shape().as_list())

	#loss
	x4=tf.reshape(x4, [2,120])
	x3=tf.reshape(x3, [10,120])
	# define the value of |xtest| and |xsupport|,x4_abs,x3_norm
	x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))  # 平方求和, axis=1, 求和120个元素, x3_norm.get_shape(): (10,)
	x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))  #x4_norm.get_shape(): (2,)
	x3_norm = tf.reshape(x3_norm, [10, 1])
	x4_abs = tf.reshape(x4_norm, [1, 2])
	#define the multiply of xtest and xsupp, xt and x3
	xt = tf.transpose(x4, perm=[1, 0])  # 切换x4的维度顺序, xt.get_shape(): (120, 2)
	xt_xs = tf.matmul(x3, xt)  # xt_xs.get_shape(): (10, 2)
	#calculate the cosin
	cosinall = xt_xs/(x3_norm*x4_abs)  # cosinall.get_shape(): (10, 2)
	# print(cosinall.get_shape())
	cosin_soft = tf.nn.softmax(cosinall, axis=0)  # cosin_soft.get_shape(): (10, 2)
	#calculate teh pre_y
	cosin_soft = tf.transpose(cosin_soft, perm=[1,0])  # (2, 10)
	# print(cosin_soft.get_shape())
	y3 = tf.reshape(y3,[1,10])  # (1, 10)
	pre_y = tf.reduce_sum(cosin_soft*y3, axis=1)  # 对应维度相乘，pre_y.get_shape(): (2, 10)
	# print((cosin_soft*y3).get_shape())
	# print(pre_y.get_shape())
	cost = tf.reduce_mean(tf.square(y4 - pre_y))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	# print(max.get_shape().as_list())

		#train
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		y_p = []
		for i in range(1):
			df = pd.read_csv("./data.csv")
			x = np.random.randint(0, 1503, size=12)
			xreal = df.ix[x, 1:]
			xarr = np.array(xreal)
			yreal = df.ix[x, 0]
			yall = np.array(yreal)
			sess.run(optimizer, feed_dict={ _X:xarr, _Y:yall})
			#print(sess.run(pre_y))
			# print(sess.run(x4))
			print(sess.run(xallhaha,feed_dict={ _X:xarr, _Y:yall}))
			y_p.append(sess.run(cost,feed_dict={ _X:xarr, _Y:yall}))
	x_p = range(1)
	plt.plot(x_p, y_p)
	plt.show()

