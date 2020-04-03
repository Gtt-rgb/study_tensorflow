import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data
#DATA
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#pre prepare
#权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#network input
xs = tf.placeholder(tf.float32, [None, 784])#/255. # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])#转换维度[全部例子，图片长，图片宽，图片通道个数]

#network conv2
W_conv1 = weight_variable([5,5, 1,32]) #[卷积核长，卷积核宽，输入图片通道个数，输出图片通道个数]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #28x28x32
h_pool1 = max_pool_2x2(h_conv1)#14*14*32

W_conv2 = weight_variable([5,5, 32,64]) #[卷积核长，卷积核宽，输入图片通道个数，输出图片通道个数]
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #14x14x64
h_pool2 = max_pool_2x2(h_conv2)#7*7*64
#平坦层
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#-1代表全部例子

#network hidden layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#compile network
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

#Evaluation network
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
#printf accu
print(compute_accuracy(
    mnist.test.images[:1000], mnist.test.labels[:1000]))