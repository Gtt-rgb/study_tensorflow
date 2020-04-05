import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data
#data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#parameter
lr = 0.001#学习率
training_iters = 100#训练轮数
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
#RNN
def RNN(X, weights, biases):#输入，权重，偏置
    X = tf.reshape(X, [-1, 28])#(128 * 28, 28 )
    X_in = tf.matmul(X, weights['in']) + biases['in']#(128 * 28, 128 )
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])#(128 ，28, 128 )#时间序列
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)#time_major代表时间序列
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']#应该是仅仅考虑了短期记忆
    return results

#input
xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])
#network
pred = RNN(xs, weights, biases)
#compile network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#train
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(training_iters):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys})
        if i==50:
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))#计算是否相等
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#计算错误率
            print(sess.run(accuracy, feed_dict={
                xs: batch_xs,
                ys: batch_ys,
                }))
