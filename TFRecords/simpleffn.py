"""

"""
import tensorflow as tf
from collections import OrderedDict
from TFRecords.rztutils import TFRecords

metadata = OrderedDict(string=dict(id=1), float_list=dict(output=3, input=4))
tfrecords_path = '/Users/umesh/PycharmProjects/Tensorflow_TFRecords/TFRecords/irisdata_TFRecords/'
epochs = 1
batch_size = 15
utils = TFRecords()
'''
This function will be useful when you have csv file and
you want to convert csv file to TFRecord and use it
'''
data = utils.convert_and_read_data_from_tfrecords(
    "/Users/umesh/PycharmProjects/Tensorflow_TFRecords/datasets/irisdata.csv", delimiter=',', metadata=metadata,
    output_label=-3,
    tfrecords_path=tfrecords_path, header=True, batch_size=batch_size,
    num_of_epochs=epochs, index_col=True,
    shuffle_batch_threads=1, capacity=2,
    min_after_deque=0,
    allow_small_final_batch=True)

'''
This function will be used when we already have an TFRecord file and we can use it directly
'''
# data = utils.setup_data_read(tfrecords_path + 'irisdata.tfrecords', batch_size=batch_size,
#                              num_of_epochs=epochs, metadata=metadata)

learning_rate = 0.01
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

init = tf.local_variables_initializer()
global_variables = tf.global_variables_initializer()

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

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
            print(data_['input'][0])
            exit()
            training_cost, optimizer, summary, acc = sess.run([cost, train_step, merged_summary_op, accuracy],
                                                              feed_dict={input_data: data_['input'],
                                                                         output_data: data_['output']})
            print("Epoch", i, "Cost", training_cost, "Accuracy:", acc)
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
        coord.join(threads)
