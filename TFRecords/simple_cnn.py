"""
@Created on: 27/09/16,
@author: Prathyush SP,
@version: 0.0.1


Sphinx Documentation:

"""

import tensorflow as tf
from TFRecords.rztutils import TFRecords
from collections import OrderedDict

metadata = OrderedDict(float_list=dict(output=10, input=784))
tfrecords_path = '/Users/umesh/PycharmProjects/Tensorflow_TFRecords/TFRecords/tfrecords/'
epochs = 10
batch_size = 250
utils = TFRecords()
data = utils.setup_data_read(tfrecords_path=tfrecords_path + 'simple_cnn_data.tfrecords', batch_size=batch_size,
                             metadata=metadata,
                             num_of_epochs=epochs,
                             shuffle_batch_threads=1, capacity=2,
                             min_after_deque=0,
                             allow_small_final_batch=True)
#
#
# data = utils.convert_and_read_data_from_tfrecords(
#     "/Users/umesh/PycharmProjects/Tensorflow_TFRecords/datasets/simple_cnn_data.csv", delimiter=';', metadata=metadata,
#     output_label=10,label_vector=True,
#     tfrecords_path=tfrecords_path, batch_size=batch_size,
#     num_of_epochs=epochs,
#     shuffle_batch_threads=1, capacity=2,
#     min_after_deque=0,
#     allow_small_final_batch=True)

# Parameters
learning_rate = 0.01
epoch = 10
batch_size = 128
display_step = 10
logs_path = 'cnn_for_mnist'

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([3136, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([10]))
}

# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder("float")
x = tf.placeholder("float", name='Input')
y = tf.placeholder("float", name='Output')


def model(x, weights, biases, dropout):
    layer1 = tf.reshape(x, shape=[-1, 28, 28, 1])
    layer2 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer1, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME'), biases['bc1']))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer4 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(layer3, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME'), biases['bc2']))
    layer5 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer5 = tf.reshape(layer5, [-1, 3136])
    layer6 = tf.nn.relu(tf.add(tf.matmul(layer5, weights['wd1']), biases['bd1']))
    layer6 = tf.nn.dropout(layer6, dropout)
    layer7 = tf.add(tf.matmul(layer6, weights['out']), biases['out'])
    return layer7


pred = model(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.local_variables_initializer()
global_variables = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     step, batch = 0, 0
#     for i in range(epoch):
#         while step < len(train_data) / batch_size:
#             batch_x, batch_y = train_data[batch: batch + batch_size], train_label[batch:batch + batch_size]
#             batch += batch_size
#             sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                            keep_prob: 0.75})
#             if epoch % display_step == 0:
#                 loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                                   y: batch_y,
#                                                                   keep_prob: 1.})
#                 print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
#                       "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                       "{:.5f}".format(acc))
#             step += 1
#     print("Optimization Finished!")
#     print("Testing Accuracy:",
#           sess.run(accuracy, feed_dict={x: train_data[:256],
#                                         y: train_label[:256],
#                                         keep_prob: 1.}))


with tf.Session() as sess:
    sess.run(init)
    sess.run(global_variables)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    try:
        while not coord.should_stop():
            i += 1
            data_ = utils.next_batch(sess, data, metadata)
            training_cost, opt, acc = sess.run([cost, optimizer, accuracy],
                                               feed_dict={x: data_['input'],
                                                          y: data_['output'], keep_prob: 0.75})
            print("Epoch", i, "Cost", training_cost, "Accuracy:", acc)
    except tf.errors.OutOfRangeError:
        print("Queue is empty")
    finally:
        coord.request_stop()
        coord.join(threads)
