import csv, os, glob
from collections import OrderedDict

import tensorflow as tf
from abc import ABCMeta, abstractmethod


class TFRecordGenerator(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def read_data(self, **args):
        """ Used for reading the data files """
        pass

    @abstractmethod
    def convert_to_tfrecord(self, **args):
        """ convert the raw data to tfrecords """
        pass

    @abstractmethod
    def read_tfrecord(self, **args):
        """ Used for reading the tfrecord data """
        pass

    """ All possible types of feature data which can be converted to tfrecord """

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def int64_list_feature(value_list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

    @staticmethod
    def float_list_feature(value_list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Generator(TFRecordGenerator):
    @staticmethod
    def choose_write_feature_func(type_):
        """
        Choose appropriate method to convert the column value into tfrecord feature based on the column type
        :param type_:
        :return:
        """
        if type_ == 'int':
            return TFRecordGenerator.int64_feature
        elif type_ == 'float':
            return TFRecordGenerator.float_feature
        elif type_ == 'intlist':
            return TFRecordGenerator.int64_list_feature
        elif type_ == 'float_list':
            return TFRecordGenerator.float_list_feature
        elif type_ == 'string':
            return TFRecordGenerator.bytes_feature

    @staticmethod
    def choose_read_feature_func(type_, data_points_length=None):
        """
        Choose appropriate method to convert the tfrecord feature into column value based on the column type
        :param type_:
        :return:
        """
        if type_ == 'int':
            return tf.FixedLenFeature([], tf.int64)
        elif type_ == 'float':
            return tf.FixedLenFeature([], tf.float32)
        elif type_ == 'intlist':
            return tf.FixedLenFeature([data_points_length], tf.int64)
        elif type_ == 'float_list':
            return tf.FixedLenFeature([data_points_length], tf.float32)
        elif type_ == 'string':
            return tf.FixedLenFeature([], tf.string)

    def convert_to_tfrecord(self, tfrecords_directory, record_name, columns_metadata, data_set):
        """
        Converts the dataset to tfrecords. Writes the records into specified directory path
        :param tfrecords_directory: directory to save tfrecords
        :param record_name: name of the tfrecord
        :param columns_metadata:
        :param data_set: type > list of objects, this is the dataset to be converted into tfrecords
        :return:
        """
        record_filepath = os.path.join(tfrecords_directory, record_name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(record_filepath)

        for data_ in data_set:
            features_ = OrderedDict()
            for type_ in columns_metadata:
                for each_key in columns_metadata[type_]:
                    val = getattr(data_, each_key)
                    if type_ == 'string':
                        val = str.encode(val)
                    features_[each_key] = self.choose_write_feature_func(type_)(val)
            example = tf.train.Example(features=tf.train.Features(feature=features_))
            writer.write(example.SerializeToString())
        writer.close()

    def read_data(self, file_path, delimiter=',', output_label_start_index=None):
        return self.read_data_from_csv(file_path, delimiter, output_label_start_index)

    @staticmethod
    def read_data_from_csv(file_path, delimiter=',', output_label_start_index=10):
        """
        Read the raw data from the file_path given and return the dataset needed to create tfrecords
        :param file_path: filepath of raw data
        :param delimiter: delimiter for column values
        :param output_label_start_index: index where output label starts
        :return:
        """
        total_samples = 0
        data_set = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for m, line in enumerate(reader):
                total_samples += 1
                line_data = [float(val) for i, val in enumerate(line)]
                data = Data(line_data[:output_label_start_index], line_data[output_label_start_index:])
                data_set.append(data)
        return data_set, total_samples

    def read_tfrecord(self, **args):
        pass

    def read_and_decode_single_example(self, file_path, num_of_epochs, columns_metadata):
        """
        Read and decode the tfrecord
        :param file_path:
        :param num_of_epochs: num of epochs
        :param columns_metadata: columns metadata
        :return:
        """
        filenames = glob.glob(file_path + os.sep + '*.tfrecords')
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_of_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features_ = OrderedDict()
        for type_ in columns_metadata:
            for each_key in columns_metadata[type_]:
                features_[each_key] = self.choose_read_feature_func(type_, data_points_length=columns_metadata[type_][
                    each_key])
        features = tf.parse_single_example(serialized_example, features=features_)
        return features, list(features_.keys())

    def setup_data_read(self, file_path, batch_size, shuffle_batch_threads, capacity, min_after_deque,
                        allow_small_final_batch, num_of_epochs, metadata):
        """
        Setup the data read controller
        :param file_path: directory path where list of tfrecords are saved
        :param batch_size: samples in each batch
        :param shuffle_batch_threads: number of threads to use
        :param capacity: capacity of the queue
        :param min_after_deque: minimum samples after deque
        :param allow_small_final_batch: allow_small_final_batch
        :param num_of_epochs: num of epochs of data needed
        :return:
        """
        features, key_list_ = self.read_and_decode_single_example(file_path, num_of_epochs, metadata)
        fetch_features = [features[key] for key in key_list_]
        features_batch = tf.train.shuffle_batch(
            fetch_features,
            batch_size=batch_size,
            num_threads=shuffle_batch_threads,
            capacity=capacity,
            min_after_dequeue=min_after_deque,
            allow_smaller_final_batch=allow_small_final_batch,
            name='BatchOp')
        return features_batch

    def next_batch(self, session, features_batch, metadata):
        """
        Call this function repeatedly to fetch all the data. (Number of calls depends on the batch size and total data size)
        :param session: TF session
        :param features_batch: list of features which can be read batchwise
        :return:
        """
        processed_data = OrderedDict()
        data_ = session.run(features_batch)
        count = 0
        for type_ in metadata:
            for each_key in metadata[type_]:
                processed_data[each_key] = data_[count] if type_ != 'string' else [bytes.decode(d) for d in
                                                                                   data_[count]]
                count += 1
        return processed_data


class Data:
    """
    Data object for creating the tfrecords
    """

    def __init__(self, input, output):
        self.input = input
        self.output = output

#
# if __name__ == '__main__':
#     g = Generator()
#     metadata = OrderedDict(float_list=dict(output=3, input=4))
#     for file_ in glob.glob('irisdata.csv'):
#         data_set, total_lines = g.read_data(file_path=file_, delimiter=',',
#                                             output_label_start_index=-3)
#         tfrecords_path = '/Users/umesh/PycharmProjects/TensorFlow_Python/TFRecords/tfrecords/'
#         g.convert_to_tfrecord(tfrecords_directory=tfrecords_path,
#                               record_name=os.path.split(file_)[1].replace('.csv', ''),
#                               columns_metadata=metadata, data_set=data_set)
#         batch_size = len(data_set)
#         batch_data = g.setup_data_read(file_path=tfrecords_path, batch_size=batch_size, shuffle_batch_threads=4, capacity=10,
#                                        min_after_deque=0, allow_small_final_batch=True, num_of_epochs=None)
#
#         sess = tf.Session()
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         tf.train.start_queue_runners(sess=sess)
#         for i in range(int(len(data_set)/batch_size)):
#             print('i>', i)
#             data_ = g.next_batch(sess, batch_data)
#             for key in data_:
#                 print(key, data_[key])
