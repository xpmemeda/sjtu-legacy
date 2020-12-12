'''
功能：训练并保存MultilRNN模型 ---多步输入、多步输出---直接预测---MMD迁移学习---
编辑时间：20191212
运行时间：11分钟 --- +mmd：12分钟 ---
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import sys
import os

# 训练参数
lr_default = 0.01 # 学习率
batch_size_default = 80 # 批大小
in_step = 10 # 输入步长
out_step = 1  # 输出步长
epoch_default = 1501 # 训练周期
# 迁移选项
mmd = 1 # 是否迁移
# p, q: mmd和L2的损失占比
p, q = 0, 0.3 
# 模型名称：关系到存放模型的目录
file_name = "multirnn"
# 源域和目标域
# path_source = "dataset/20170928_51-53.csv"
# path_target = "dataset/20171017_51-53.csv"
path_source = "dataset/data_2.npy" # data_2(7:40), data_3(8:10)
path_target = "dataset/data_3.npy" 

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

def get_data(in_step=in_step, path=None):
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
    for i in range(in_step+out_step, dataset.shape[0]):
        x = np.append(x, dataset[i - in_step - out_step:i - out_step])
        y = np.append(y, dataset[i - out_step:i])
    x = np.reshape(x, (-1, in_step, size))
    y = np.reshape(y, (-1, out_step, size))

    # 返回 X, Y 和 Min, Max 和 size
    return x, y, Max, Min, size

def net(x):
    # pdb.set_trace()
    # 得到输入数据的维度 batch_size, in_step, size
    batch_size = tf.shape(x)[0]  # Tensor("strided_slice:0", shape=(), dtype=int32)
    in_step, size = map(int, np.shape(x)[1:])

    # FC1:
    # with tf.variable_scope("FC1"):
    input_fc1 = tf.reshape(x, (-1, size))
    w_fc1 = tf.Variable(tf.random_normal((size, 32), seed=0), name='w')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=(1, 32)), name='b')
    wx_plus_b_L1 = tf.add(tf.matmul(input_fc1, w_fc1), b_fc1)
    fc1 = tf.nn.tanh(wx_plus_b_L1)

    # MultiRNN：
    # with tf.variable_scope("MultiRNN"):
    input_MultiRNN = tf.reshape(fc1, (-1, in_step, 32))  # shape=(?, 10, 3)
    MultiRNN_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(num_units=32) for _ in range(3)])  # 生成 MultiRNN_Cell， 三层
    initial_state_MultiRNN = MultiRNN_cell.zero_state(batch_size, np.float32)  # 初始化
    output_MultiRNN, final_state_MultiRNN = tf.nn.dynamic_rnn(MultiRNN_cell, input_MultiRNN, initial_state=initial_state_MultiRNN)
    tf.add_to_collection("output_mid", output_MultiRNN) 
    # output_MultiRNN --> Tensor("MultiRNN/rnn/transpose_1:0", shape=(?, 10, 50), dtype=float32)，代表每个 step 的输出
    # final_state 是一个 tuple，每个元素表示每一层 RNN_Cell 执行完之后的隐藏层状态，维度是 (?, 50)

    # FC2:
    # with tf.variable_scope("FC2"):
    input_fc2 = output_MultiRNN[:, -1, :]
    w_fc2 = tf.Variable(tf.random_normal((32, size), seed=0), name="w")  # 生成 W_Fc2，同时初始化
    b_fc2 = tf.Variable(tf.constant(0.1, shape=(1, size)), name="b")  # 生成 b_Fc2，同时初始化
    wx_plus_b_L2 = tf.add(tf.matmul(input_fc2, w_fc2), b_fc2)
    fc2 = tf.nn.sigmoid(wx_plus_b_L2)

    # 继续前进
    prediction = tf.reshape(fc2, (-1, 1, size))
    for i in range(out_step - 1):
        wx_plus_b_fc1 = tf.add(tf.matmul(fc2, w_fc1), b_fc1)
        fc1 = tf.nn.tanh(wx_plus_b_fc1)
        MultiRNN, final_state_MultiRNN = MultiRNN_cell.call(fc1, final_state_MultiRNN)
        wx_plus_b_fc2 = tf.add(tf.matmul(MultiRNN, w_fc2), b_fc2)
        fc2 = tf.nn.sigmoid(wx_plus_b_fc2)
        prediction = tf.concat([prediction, tf.reshape(fc2, (-1, 1, size))], 1)

    return output_MultiRNN[:, -1, :], prediction

def train(lr=lr_default, batch_size=batch_size_default, in_step=in_step, train_start=0, train_end=7000,
          test_start=7000, test_end=10000):
    # 重置 graph
    tf.reset_default_graph()
    # 获取数据
    x_s, y_s, Max_s, Min_s, size_s = get_data(in_step=in_step, path=path_source)
    x_t, y_t, max_t, min_t, size_t = get_data(in_step=in_step, path=path_target)
    # 生成 batch_index
    batch_index_train = get_batch_index(train_start, train_end, batch_size)
    batch_index_test = get_batch_index(test_start, test_end, batch_size)

    # 定义输入 placeholder
    x_ = tf.placeholder(tf.float32, shape=[None, in_step, size_s], name="input_s")
    y_ = tf.placeholder(tf.float32, shape=[None, out_step, size_s], name="label_s")
    mid_t_ = tf.placeholder(tf.float32, shape=[None, 32], name="mid_t")
    # 获得输出 prediction
    mid, pred = net(x_)

    # train_op
    loss_label = tf.sqrt(tf.reduce_mean(tf.square(pred - y_)))
    
    # 差异损失函数和正则项
    from mmd import mix_rbf_mmd2
    loss_mmd = tf.sqrt(mix_rbf_mmd2(mid, mid_t_)) 
    L2 = tf.sqrt(tf.reduce_mean(tf.square(mid)))

    # 是否迁移
    if (mmd):
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_label + p * loss_mmd + q * L2) # 最佳的参数 1：0.6：0.8
    else:
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_label)
    
    # 收藏输入输出
    tf.add_to_collection("input", x_)
    tf.add_to_collection("input", y_)
    tf.add_to_collection("predict", pred)
    # 保存模型
    saver = tf.train.Saver(max_to_keep=4)

    np_loss_train = []
    np_loss_test = []
    np_loss_target = []

    with tf.Session() as sess:
        # tensorboard 存图
        if os.path.exists("tensorboard\\"+file_name):
            del_dir("tensorboard\\"+file_name)
        train_writer = tf.summary.FileWriter("tensorboard/"+file_name+"/train/", sess.graph)
        test_writer = tf.summary.FileWriter("tensorboard/"+file_name+"/test/", sess.graph)
        # 初始化参数
        sess.run(tf.global_variables_initializer())

        # # 计算中间层输出
        # mid_t = sess.run(mid, feed_dict={x_:x_t})
        # 打印最初的损失
        loss_train = 0.0
        loss_test = 0.0
        loss_target = 0.0
        for step in range(len(batch_index_train) - 1):
            loss_train_tmp = sess.run(loss_label, feed_dict={x_: x_s[batch_index_train[step]:batch_index_train[step + 1]],
                                                         y_: y_s[batch_index_train[step]:batch_index_train[step + 1]]})
            loss_train = loss_train + (batch_index_train[step + 1] - batch_index_train[step]) * loss_train_tmp
        for step in range(len(batch_index_test) - 1):
            loss_test_tmp = sess.run(loss_label, feed_dict={x_: x_s[batch_index_test[step]:batch_index_test[step + 1]],
                                                        y_: y_s[batch_index_test[step]:batch_index_test[step + 1]]})
            loss_test = loss_test + (batch_index_test[step + 1] - batch_index_test[step]) * loss_test_tmp
        for step in range(len(batch_index_train) - 1):
            loss_target_tmp = sess.run(loss_label, feed_dict={x_: x_t[batch_index_train[step]:batch_index_train[step + 1]],
                                                         y_: y_t[batch_index_train[step]:batch_index_train[step + 1]]})
            loss_target = loss_target + (batch_index_train[step + 1] - batch_index_train[step]) * loss_target_tmp

        L2_p = sess.run(L2, feed_dict={x_: x_s[batch_index_train[0]:batch_index_train[1]]})
        loss_train = loss_train / (batch_index_train[-1] - batch_index_train[0])
        loss_test = loss_test / (batch_index_test[-1] - batch_index_test[0])
        loss_target = loss_target / (batch_index_train[-1] - batch_index_train[0])
        print("初始值：", L2_p)
        print('loss_train: ' + str(loss_train) + '， loss_test: ' + str(loss_test))
        print('loss_target: ' + str(loss_target))
        
        # 记录数值：论文做图
        np_loss_train.append(loss_train)
        np_loss_test.append(loss_test)
        np_loss_target.append(loss_target)

        # 开始训练
        t1 = datetime.datetime.now()
        for epoch in range(epoch_default):
            for step in range(len(batch_index_train) - 1):
                mid_t = sess.run(mid, feed_dict={x_:x_t[batch_index_train[0]:batch_index_train[1]]}) # reduce the samples in target domain. 
                sess.run(train_op, feed_dict={x_: x_s[batch_index_train[step]:batch_index_train[step + 1]],
                                              y_: y_s[batch_index_train[step]:batch_index_train[step + 1]],
                                              mid_t_: mid_t})
            if (epoch % 10 == 0):
                loss_train = 0.0
                loss_test = 0.0
                loss_target = 0.0
                for step in range(len(batch_index_train) - 1):
                    loss_train_tmp = sess.run(loss_label, feed_dict={x_: x_s[batch_index_train[step]:batch_index_train[step + 1]],
                                                                 y_: y_s[batch_index_train[step]:batch_index_train[step + 1]]})
                    loss_train = loss_train + (batch_index_train[step + 1] - batch_index_train[step]) * loss_train_tmp
                for step in range(len(batch_index_test) - 1):
                    loss_test_tmp = sess.run(loss_label, feed_dict={x_: x_s[batch_index_test[step]:batch_index_test[step + 1]],
                                                                y_: y_s[batch_index_test[step]:batch_index_test[step + 1]]})
                    loss_test = loss_test + (batch_index_test[step + 1] - batch_index_test[step]) * loss_test_tmp
                for step in range(len(batch_index_train) - 1):
                    loss_target_tmp = sess.run(loss_label, feed_dict={x_: x_t[batch_index_train[step]:batch_index_train[step + 1]],
                                                                 y_: y_t[batch_index_train[step]:batch_index_train[step + 1]]})
                    loss_target = loss_target + (batch_index_train[step + 1] - batch_index_train[step]) * loss_target_tmp

                L2_p = sess.run(L2, feed_dict={x_: x_s[batch_index_train[0]:batch_index_train[1]]})
                loss_train = loss_train / (batch_index_train[-1] - batch_index_train[0])
                loss_test = loss_test / (batch_index_test[-1] - batch_index_test[0])
                loss_target = loss_target / (batch_index_train[-1] - batch_index_train[0])
                print("epoch " + str(epoch) + ":", L2_p)
                print('loss_train: ' + str(loss_train) + '， loss_test: ' + str(loss_test))
                print('loss_target: ' + str(loss_target))
                
                # 记录数值：论文做图
                np_loss_train.append(loss_train)
                np_loss_test.append(loss_test)
                np_loss_target.append(loss_target)

                saver.save(sess, "models/"+file_name+"/"+file_name, global_step=epoch)
                print("save the model")
        '''
        # 打印最后的output_mid，是否出现特征消失和梯度爆炸
        final_mid = sess.run(mid, feed_dict={x_:x_s})
        print(final_mid)
        '''
        
        # 保存训练过程中的损失函数值：论文作图
        np_loss_train = np.array(np_loss_train)
        np_loss_test = np.array(np_loss_test)
        np_loss_target = np.array(np_loss_target)
        np.save("loss_train.npy", np_loss_train)
        np.save("loss_test.npy", np_loss_test)
        np.save("loss_target.npy", np_loss_target)
        
        # 保存训练集最后的预测值和真实值：论文作图
        prediction_train_set = sess.run(pred, feed_dict={x_: x_s[batch_index_train[0]:batch_index_train[1]]})
        label_train_set = y_s[batch_index_train[0]:batch_index_train[1]]
        np.save("prediction_train_set.npy", prediction_train_set)
        np.save("label_train_set.npy", label_train_set)
        # 保存测试集最后的预测值和真实值：论文作图
        prediction_test_set = sess.run(pred, feed_dict={x_: x_s[batch_index_test[0]:batch_index_test[1]]})
        label_test_set = y_s[batch_index_test[0]:batch_index_test[1]]
        np.save("prediction_test_set.npy", prediction_test_set)
        np.save("label_test_set.npy", label_test_set)
        # 保存目标域最后的预测值和真实值：论文作图
        prediction_target_set = sess.run(pred, feed_dict={x_: x_t[batch_index_test[0]:batch_index_test[1]]})
        label_target_set = y_t[batch_index_test[0]:batch_index_test[1]]
        np.save("prediction_target_set.npy", prediction_target_set)
        np.save("label_target_set.npy", label_target_set)
        
        t2 = datetime.datetime.now()
        print(t2 - t1)


if (__name__ == "__main__"):

    train(train_start=0, train_end=3000, test_start=3000, test_end=9900)
