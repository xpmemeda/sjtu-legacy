'''
功能：加载之前训练出来的模型，用作新的预测
编辑时间：20191212
执行时间：17秒
'''

import tensorflow as tf
import numpy as np 
import pandas as pd 
import datetime
import sys

lr_default = 0.01
batch_size_default = 80
time_step_default = 10
epoch_default = 101


def get_batch_index(start, end, batch_size = batch_size_default):
    batch_index = []
    n_batch = (end - start) // batch_size + 1
    for i in range(n_batch + 1):
        if (i == n_batch):
            batch_index.append(end)
        else:
            batch_index.append(start + batch_size * i)
    return batch_index

def get_data(time_step = time_step_default):
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
    # 归一化 Min_Max
    Max = [np.max(dataset[:, i]) for i in range(size)]
    Min = [np.min(dataset[:, i]) for i in range(size)]
    for i in range(size):
        dataset[:, i] = (dataset[:, i] - Min[i]) / (Max[i] - Min[i])
    # 生成训练数据 X 和标签 Y
    x = np.array([])
    y = np.array([])
    for i in range(time_step, dataset.shape[0]):
        x = np.append(x, dataset[i-time_step:i])
        y = np.append(y, dataset[i])
    x = np.reshape(x, (-1, time_step, size))
    y = np.reshape(y, (-1, size))
    # 函数返回 X, Y, Min, Max, size
    return x, y, Max, Min, size

def train(lr = lr_default, batch_size = batch_size_default, time_step = time_step_default, train_start = 0, train_end = 7000, test_start = 7000, test_end = 10000):
    # 清空 graph
    tf.reset_default_graph()

    # 加载数据
    x, y, Max, Min, size = get_data(time_step = time_step)
    batch_index_train = get_batch_index(train_start, train_end, batch_size)
    batch_index_test = get_batch_index(test_start, test_end, batch_size)

    # 定义 session
    with tf.Session() as sess:
        # 加载 graph 和 weights
        saver = tf.train.import_meta_graph("models/" + model_name + "/" + model_name + "-400.meta") # load graph
        saver.restore(sess, tf.train.latest_checkpoint("models/" + model_name + "/")) # load weights
        graph = tf.get_default_graph()
        # 已经加载训练好的参数，不用再初始化
        # sess.run(tf.global_variables_initializer())

        # 得到 graph 中的 tensor，方便对其操作，可以通过 get_collection() 获得，也可以通过名字
        # 输入
        x_, y_ = tf.get_collection("input")
        # x_ = graph.get_tensor_by_name("input:0")
        # y_ = graph.get_tensor_by_name('label:0')
        # 中间值
        # tmp = tf.get_collection("tmp")[0]
        output_mid_ = tf.get_collection("output_mid")[0] 
        # 输出
        pred = tf.get_collection("predict")[0]    
        # pred = graph.get_tensor_by_name("fc2/Sigmoid:0")

        # 重新定义 loss
        loss = tf.sqrt(tf.reduce_mean(tf.square(pred - y_)))
        # 定义参与训练的变量
        # fine_tune_variables = tf.all_variable() # 所有可训练的参数，等同于下面的 scope = ".*"
        fine_tune_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "FC2") # re.compile()
        # 重新定义 train_op
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list = fine_tune_variables)
        '''
        原 graph 中的变量均可被调用出来，包括 loss 和 train_op 
        # output = graph.get_tensor_by_name('rnn/rnn/transpose_1:0')
        # input_fc2 = graph.get_tensor_by_name('fc2/strided_slice:0') 
        # pred = graph.get_tensor_by_name('fc2/Sigmoid:0')
        # loss = graph.get_tensor_by_name('Sqrt:0')
        # train_op = graph.get_operation_by_name('GradientDescent')
        # output = tf.stop_gradient(output)
        # input_fc2 = output[:, -1, :]
        # # pred = tf.stop_gradient(pred)
        # # # fine tuning
        # # weights = tf.Variable(tf.truncated_normal((3, 3), stddev=0.1))
        # # biases = tf.Variable(tf.constant(0.1, shape=(1, 3)))
        # # plus = tf.matmul(pred, weights) + biases
        # # fc = tf.nn.sigmoid(plus)
        # # # define new loss function and train op.
        # # loss_ft = tf.sqrt(tf.reduce_mean(tf.square(fc - y_)))
        # # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_ft)
        '''
                 
        # 获得中间值
        # tmp_MultiRNN = np.array(sess.run(tmp, feed_dict = {x_:x}))
        output_mid = np.array(sess.run(output_mid_, feed_dict = {x_:x}))

        # 打印初始阶段的 loss
        loss_train = sess.run(loss, feed_dict={x_:x[train_start:train_end], y_:y[train_start:train_end]})
        loss_test  = sess.run(loss, feed_dict={x_:x[test_start:test_end], y_:y[test_start:test_end]})
        print("loss_train: " + str(loss_train) + ", loss_test: " + str(loss_test))

        # 开始训练
        print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        for epoch in range(epoch_default):
            for step in range(len(batch_index_train)-1):
                sess.run(train_op, feed_dict={output_mid_:output_mid[batch_index_train[step]:batch_index_train[step+1]], y_:y[batch_index_train[step]:batch_index_train[step+1]]})
            if (epoch % 10 == 0):
                loss_train = sess.run(loss, feed_dict={output_mid_:output_mid[train_start:train_end], y_:y[train_start:train_end]})
                loss_test  = sess.run(loss, feed_dict={output_mid_:output_mid[test_start:test_end], y_:y[test_start:test_end]})
                print("epoch: " + str(epoch) + "   loss_train: " + str(loss_train) + ", loss_test: " + str(loss_test))
        print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
if (__name__ == "__main__"):

    path = 'dataset/20170928_51-53.csv'
    path = 'dataset/data_4.npy'

    try:
        model_name = sys.argv[1]
    except:
        model_name = "multilstm"

    train(train_start = 0, train_end = 3000, test_start = 3000, test_end = 10000)
