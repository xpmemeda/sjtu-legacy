'''
功能：训练MultiLSTM模型
编辑时间：20191212
运行时间：11分钟 --- +mmd：12分钟 ---
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import sys
import os

lr_default = 0.01
batch_size_default = 80
time_step_default = 10
epoch_default = 401

gamma_st = 0.4

def del_dir(path):
    if os.path.isfile(path):
        os.remove(path)
        return
    else:
        for i in os.listdir(path):
            c_path = os.path.join(path, i)
            del_dir(c_path)
        # os.removedirs(path)

def get_batch_index(start_line, end_line, batch_size=batch_size_default):
    batch_index = []
    n_batch = (end_line - start_line) // batch_size + 1
    for i in range(n_batch + 1):
        if (i == n_batch):
            batch_index.append(end_line)
        else:
            batch_index.append(start_line + batch_size * i)
    return batch_index

def get_data(time_step=time_step_default, path=None):
    # 读取csv数据
    if (path[-3:] == "csv"):
        dataset = pd.read_csv(path, header=0)
        dataset = np.array(dataset)
        dataset = dataset[:10000]
        size = dataset.shape[-1]  # 数据的特征维度
    # 读取npy数据
    if (path[-3:] == "npy"):
        dataset = np.load(path)
        size = dataset.shape[-1]

    # 数据归一化处理 Min_Max
    Max = [np.max(dataset[:, i]) for i in range(size)]
    Min = [np.min(dataset[:, i]) for i in range(size)]
    for i in range(size):
        dataset[:, i] = (dataset[:, i] - Min[i]) / (Max[i] - Min[i])

    # 生成输入数据 X 和标签 Y
    x = np.array([])
    y = np.array([])
    for i in range(time_step, dataset.shape[0]):
        x = np.append(x, dataset[i - time_step:i])
        y = np.append(y, dataset[i])
    x = np.reshape(x, (-1, time_step, size))
    y = np.reshape(y, (-1, size))

    # 返回 X, Y 和 Min, Max 和 size
    return x, y, Max, Min, size

def net(x):
    # 得到输入数据的维度 batch_size, time_step, size
    batch_size = tf.shape(x)[0]  # Tensor("strided_slice:0", shape=(), dtype=int32)
    time_step, size = map(int, np.shape(x)[1:])

    # fc1:
    with tf.variable_scope("FC1"):
        input_fc1 = tf.reshape(x, (-1, size))
        w_fc1 = tf.Variable(tf.random_normal((size, 128), seed=0), name='w')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=(1, 128)), name='b')
        wx_plus_b_L1 = tf.add(tf.matmul(input_fc1, w_fc1), b_fc1)
        fc1 = tf.nn.tanh(wx_plus_b_L1)

    # MultiLSTM： 3层LSTM
    with tf.variable_scope("MultiLSTM"):
        input_MultiLSTM = tf.reshape(fc1, (-1, time_step, 128))  # shape=(?, 10, 3)
        MultiLSTM_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=50) for _ in range(3)])  # 生成 MultiLSTM_Cell， 三层
        initial_state_MultiLSTM = MultiLSTM_cell.zero_state(batch_size, np.float32)  # 初始化
        output_MultiLSTM, final_state_MultiLSTM = tf.nn.dynamic_rnn(MultiLSTM_cell, input_MultiLSTM, initial_state=initial_state_MultiLSTM)  # 计算结果，output 和 final_state
        tf.add_to_collection("output_mid", output_MultiLSTM) 
    # output_MultiLSTM --> Tensor("MultiRNN/rnn/transpose_1:0", shape=(?, 10, 50), dtype=float32)，代表每个 step 的输出
    # final_state 是一个 tuple，每个元素表示每一层 RNN_Cell 执行完之后的隐藏层状态，维度是 (?, 50)

    # fc2:
    with tf.variable_scope("FC2"):
        input_fc2 = output_MultiLSTM[:, -1, :]
        w_fc2 = tf.Variable(tf.random_normal((50, size), seed=0), name="w")  # 生成 W_Fc2，同时初始化
        b_fc2 = tf.Variable(tf.constant(0.1, shape=(1, size)), name="b")  # 生成 b_Fc2，同时初始化
        wx_plus_b_L2 = tf.add(tf.matmul(input_fc2, w_fc2), b_fc2)
        fc2 = tf.nn.sigmoid(wx_plus_b_L2)

    return input_fc2, fc2

def train(lr=lr_default, batch_size=batch_size_default, time_step=time_step_default, train_start=0, train_end=7000,
          test_start=7000, test_end=10000):
    # 重置 graph
    tf.reset_default_graph()

    # 获取数据

    x_s, y_s, Max_s, Min_s, size_s = get_data(time_step=time_step, path=path_source)
    x_t, y_t, max_t, min_t, size_t = get_data(time_step=time_step, path=path_target)
    # 生成 batch_index
    batch_index_train = get_batch_index(train_start, train_end, batch_size)
    batch_index_test = get_batch_index(test_start, test_end, batch_size)

    # 定义输入 placeholder
    x_ = tf.placeholder(tf.float32, shape=[None, time_step, size_s], name="input_s")
    y_ = tf.placeholder(tf.float32, shape=[None, size_s], name="label_s")
    mid_t_ = tf.placeholder(tf.float32, shape=[None, 50], name="mid_t")
    # 获得输出 prediction
    mid, pred = net(x_)
    # mid_re: Tensor("FC2/strided_slice:0", shape=(?, 50), dtype=float32)
    # pred: Tensor("FC2/Sigmoid:0", shape=(?, 3), dtype=float32)

    # train_op
    loss_s = tf.sqrt(tf.reduce_mean(tf.square(pred - y_)))
    # loss_st = tf.sqrt(tf.reduce_mean(tf.square(mid - mid_t_))) # 测试集 0.12 --> 0.174, gamma_st=0.4
    # loss_st = tf.sqrt(tf.reduce_mean(tf.square(tf.reduce_mean(mid, axis=0) - tf.reduce_mean(mid_t_, axis=0)))) # 测试集 0.12 --> 0.156, gamma_st=0.4
    from mmd import mix_rbf_mmd2
    loss_st = tf.sqrt(mix_rbf_mmd2(mid, mid_t_)) 
    L2 = tf.sqrt(tf.reduce_mean(tf.square(mid)))
    if (mmd):
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_s + p * loss_st + q * L2) # 最佳的参数 1：0.6：0.8
    else:
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_s + 0 * loss_st + 0 * L2) # 最佳的参数 1：0.6：0.8
    # train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 保存模型
    tf.add_to_collection("input", x_)
    tf.add_to_collection("input", y_)
    # tf.add_to_collection("tmp", "MultiRNN/rnn/transpose_1:0")
    tf.add_to_collection("predict", pred)
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:
        # tensorboard 存图
        if os.path.exists("tensorboard\\"+file_name):
            del_dir("tensorboard\\"+file_name)
        train_writer = tf.summary.FileWriter("tensorboard/"+file_name+"/train/", sess.graph)
        test_writer = tf.summary.FileWriter("tensorboard/"+file_name+"/test/", sess.graph)
        # 初始化参数
        sess.run(tf.global_variables_initializer())

        # 打印初始状态 loss
        mid_t = sess.run(mid, feed_dict={x_:x_t})

        loss_train = sess.run(loss_s, feed_dict={x_: x_s[train_start:train_end], y_: y_s[train_start:train_end]})
        loss_test = sess.run(loss_s, feed_dict={x_: x_s[test_start:test_end], y_: y_s[test_start:test_end]})
        loss_mmd = sess.run(loss_st, feed_dict={x_:x_s[train_start:train_end], mid_t_:mid_t[train_start:train_end]})
        loss_L2 = sess.run(L2, feed_dict = {x_:x_s[train_start:train_end]})
        print('loss_train: ' + str(loss_train) + '， loss_test: ' + str(loss_test))
        print("loss_mmd: "+str(loss_mmd))
        print("loss_L2: " + str(loss_L2))
        # 开始训练
        print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        for epoch in range(epoch_default):
            for step in range(len(batch_index_train) - 1):
                # mid_t = sess.run(mid, feed_dict={x_:x_t[batch_index_train[step]:batch_index_train[step + 1]]})
                mid_t = sess.run(mid, feed_dict={x_:x_t[batch_index_train[0]:batch_index_train[1]]}) # reduce the samples in target domain. 
                sess.run(train_op, feed_dict={x_: x_s[batch_index_train[step]:batch_index_train[step + 1]],
                                              y_: y_s[batch_index_train[step]:batch_index_train[step + 1]],
                                              mid_t_: mid_t})
            if (epoch % 10 == 0):
                mid_t = sess.run(mid, feed_dict={x_:x_t})
                loss_train = sess.run(loss_s, feed_dict={x_: x_s[train_start:train_end], y_: y_s[train_start:train_end]})
                loss_test = sess.run(loss_s, feed_dict={x_: x_s[test_start:test_end], y_: y_s[test_start:test_end]})
                loss_mmd = sess.run(loss_st, feed_dict={x_: x_s[train_start:train_end], mid_t_:mid_t[train_start:train_end]})
                loss_L2 = sess.run(L2, feed_dict = {x_:x_s[train_start:train_end]})
                print('iter: ' + str(epoch) + '   loss_train: ' + str(loss_train) + '， loss_test: ' + str(loss_test))
                print("iter: " + str(epoch) + "   loss_mmd: " + str(loss_mmd))
                print("loss_L2: " + str(loss_L2))

                saver.save(sess, "models/"+file_name+"/"+file_name, global_step=epoch)
                print("save the model")
                '''
                midprint = sess.run(mid, feed_dict={x_:x_s[batch_index_train[1]:batch_index_train[2]]})
                print(midprint)
                '''
        # final_mid = sess.run(mid, feed_dict={x_:x_s})
        # print(final_mid)
        
        '''
        # 保存最后的预测值和真实值：论文作图 ---一步预测---
        prediction = sess.run(pred, feed_dict={x_: x_s[batch_index_train[0]:batch_index_train[1]]})
        label = y_s[batch_index_train[0]:batch_index_train[1]]
        np.save("prediction.npy", prediction)
        np.save("label.npy", label)
        '''

        '''
        # 保存最后的预测值和真实值：论文作图 ---五步预测---
        prediction = sess.run(pred, feed_dict={x_: x_s[batch_index_train[0]:batch_index_train[1]]}) 
        for i in range(5):
            prediction = sess.run(pred, feed_dict={x_:prediction[0:80]})
        '''

    print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


if (__name__ == "__main__"):
    file_name = "multilstm"

    path_source = "dataset/20170928_51-53.csv"
    path_target = "dataset/20171017_51-53.csv"
    path_source = "dataset/data_2.npy" # 7:40
    path_target = "dataset/data_3.npy" # 8:10

    mmd = 0
    p, q = 0.3, 0.8

    train(train_start=0, train_end=3000, test_start=3000, test_end=10000)
