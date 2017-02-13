import tensorflow as tf
from collections import OrderedDict
from TFRecords import tfrecord_utils
import os
import time

utils = tfrecord_utils.Generator()
metadata = OrderedDict(float_list=dict(output=3, input=4))
file_path = 'irisdata.csv'
train_data, no_of_records = utils.read_data(file_path=file_path, delimiter=',', output_label_start_index=-3)
tfrecords_path = '/Users/umesh/PycharmProjects/TensorFlow_Python/TFRecords/tfrecords/'
utils.convert_to_tfrecord(tfrecords_directory=tfrecords_path,
                          record_name=os.path.split(file_path)[1].replace('.csv', ''), columns_metadata=metadata,
                          data_set=train_data)
batch_size = 30
batch_display_step = 1
batch_data = utils.setup_data_read(file_path=tfrecords_path, batch_size=batch_size, shuffle_batch_threads=4,
                                   capacity=10, min_after_deque=0, allow_small_final_batch=True, num_of_epochs=None,
                                   metadata=metadata)

no_of_inputs = len(train_data)
learning_rate = 0.01
epochs = 500
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
        epoch_cost, epoch_acc = [], []
        for i in range(int(no_of_records / batch_size)):
            data = utils.next_batch(sess, batch_data, metadata)
            training_cost, optimizer, summary, acc = sess.run([cost, train_step, merged_summary_op, accuracy],
                                                              feed_dict={input_data: data['input'],
                                                                         output_data: data['output']})
            epoch_acc.append(acc)
            # if i % batch_display_step == 0:
            #     print("\t\tBatch: ", i, "cost: ", training_cost, "Accuracy: ", acc)
            epoch_cost.append(training_cost)
            summary_writer.add_summary(summary, each_epoch)
        if each_epoch % display_step == 0:
            print("Epoch", each_epoch, ":", "cost is :", sum(epoch_cost) / len(epoch_cost), "Accuracy:",
                  sum(epoch_acc) / len(epoch_acc))
    print(time.time() - start_time)
    # exit()
    coord.request_stop()
    coord.join(threads)
    sess.close()

    # acc = sess.run(accuracy, feed_dict={input_data: test_data_input, output_data: test_data_output})
    # print("Accuracy is :", acc)
