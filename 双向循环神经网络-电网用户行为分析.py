# region 载入库,配置参数
import tensorflow as tf
import numpy as np

learning_rate = 0.01    # 因为优化器是adam,所以学习速率较低
max_samples = 400000    # 最大训练样本数为40万
batch_size = 128
display_step = 10       # 每间隔10次训练,就展示一次训练情况
n_input = 28 # 图像的宽度为28,因此设置输入为28
n_steps = 28 # 图像的高度为28,因此设置LSTM的展开步数(unrolled steps of LSTM)也设置为28
n_hidden = 256 # 定义一个方向的cell的数量
n_classes = 10 # 0-9,共有10个分类.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# endregion

# region 定义softmax层的权重
# 因为要合成两个LSTM的输入,所以第一个维度是2*n_hidden
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# endregion

# region 构建计算图
# x是一个二维结构,但是和卷积网络中的空间二维结构不同,
# 这里的二维被理解成第一个维度是时间序列n_steps,第二维度是每个时间点下的数据n_input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
# 将x拆成一个长度为n_steps的列表,每个元素tensor的尺寸为[batch_size,n_input]
x_unstack = tf.unstack(x, axis=1)

# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_bw_cell = tf.contrib.rnn.GRUCell(n_hidden)

outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_unstack,
                                           dtype=tf.float32)
pred=tf.matmul(outputs[-1], weights['out']) + biases['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# tf.argmax(pred,1)求每行中最大的元素的角标(即列号)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# tf.cast(correct_pred, tf.float32),将correct_pred转化为浮点型
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
# endregion

# region 执行计算图
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:# step*128<400000
        # 直接读出来的batch_x的尺寸为[batch_size,784]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x经过reshape后,尺寸变为(batch_size, n_steps, n_input)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
# endregion