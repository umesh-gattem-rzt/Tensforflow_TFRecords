import tensorflow as tf
from collections import OrderedDict
from TFRecords import tfrecord_utils
import os
import time
from TFRecords import tfrecord_utils, rztutils

util = tfrecord_utils.TFRecords()
metadata = OrderedDict(float_list=OrderedDict(data=7))
file_path = 'irisdata.csv'
data, no_of_records = util.read_data_from_csv(filepath=file_path, delimiter=',')

tfrecords_path = '/Users/umesh/PycharmProjects/Tensorflow_TFRecords/TFRecords/tfrecords/'
util.convert_to_tfrecord(tfrecords_directory=tfrecords_path,
                         record_name=os.path.split(file_path)[1].replace('.csv', ''),
                         columns_metadata=metadata,
                         data_set=data)

training_data = util.read_data_from_tfrecord(file_path=tfrecords_path + 'irisdata.tfrecords', metadata=metadata,
                                             batch_size=30, shuffle_batch_threads=5, capacity=12,
                                             min_after_dequeue=0,
                                             allow_small_final_batch=True, num_of_epochs=1)

data_list = util.change_tensor_data_to_list(training_data)
train_data, train_label = rztutils.read_csv(data=data_list,
                                            split_ratio=[100, 0, 0],
                                            delimiter=";",
                                            normalize=False,
                                            randomize=True,
                                            output_label=[4, 5, 6])
learning_rate = 0.01
epochs = 1000
display_step = 100
logs_path = 'ffn_for_iris_data'

input_data = tf.placeholder("float", name='Input')
output_data = tf.placeholder("float", name='Output')

weights = {
    'weight1': tf.Variable(tf.random_normal([4, 7], dtype=tf.float32), name='Weight1'),
    'weight2': tf.Variable(tf.random_normal([7, 6], dtype=tf.float32), name='Weight2'),
    'weight3': tf.Variable(tf.random_normal([6, 3], dtype=tf.float32), name='Weight3'),
}

bias = {
    'bias1': tf.Variable(tf.random_normal([7]), dtype=tf.float32, name='Bias1'),
    'bias2': tf.Variable(tf.random_normal([6]), dtype=tf.float32, name='Bias2'),
    'bias3': tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='Bias3')
}


def model(x, weights, bias):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weight1']), bias['bias1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
    output_layer = tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3'])
    return output_layer


with tf.name_scope('Activation'):
    activation = model(input_data, weights, bias)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.square(output_data - activation))

with tf.name_scope('Optimization'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(activation, 1), tf.argmax(output_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    for each_epoch in range(epochs):
        training_cost, optimizer, summary, acc = sess.run([cost, train_step, merged_summary_op, accuracy],
                                                          feed_dict={input_data: train_data,
                                                                     output_data: train_label})
        if each_epoch % display_step == 0:
            print("Epoch", each_epoch, "Cost", training_cost, "Accuracy:", acc)
    print(time.time() - start_time)
    coord.request_stop()
    coord.join(threads)
    sess.close()
