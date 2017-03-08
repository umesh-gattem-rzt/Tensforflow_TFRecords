import tensorflow as tf
import csv
import os, glob
from collections import OrderedDict


class TFRecords(object):
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

    def choose_write_feature_func(self, type_):
        """
        Choose appropriate method to convert the column value into tfrecord feature based on the column type
        :param type_:
        :return:
        """
        if type_ == 'int':
            return self.int64_feature
        elif type_ == 'float':
            return self.float_feature
        elif type_ == 'intlist':
            return self.int64_list_feature
        elif type_ == 'float_list':
            return self.float_list_feature
        elif type_ == 'string':
            return self.bytes_feature

    @staticmethod
    def choose_read_feature_func(type_, data_points_length=None):
        """
        Choose appropriate method to convert the tfrecord feature into column value based on the column type
        :param type_:
        :param data_points_length
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

    @staticmethod
    def read_data_from_csv(filepath, delimiter=','):
        """
        Read the raw data from the file_path given and return the data needed to create tfrecords
        :param filepath: file path to read raw data
        :param delimiter: delimiter for column values
        :return: data and no of records
        """
        total_samples = 0
        data_set = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                total_samples += 1
                line_data = [float(val) for val in line]
                data_set.append(line_data)
        return data_set, total_samples

    def convert_to_tfrecord(self, tfrecords_directory, record_name, columns_metadata, data_set):
        """
        Convert the dataset to tfrecords of specified directory.
        :param tfrecords_directory: directory to save tfrecords
        :param record_name: name of the tfrecord
        :param columns_metadata: metadata
        :param data_set: dataset to be converted to the tfrecord
        :return:
        """
        record_filepath = os.path.join(tfrecords_directory, record_name + '.tfrecords')
        if os.path.exists(record_filepath):
            return
        else:
            writer = tf.python_io.TFRecordWriter(record_filepath)
            for data_ in data_set:
                features_ = OrderedDict()
                for type_ in columns_metadata:
                    for each_key in columns_metadata[type_]:
                        val = data_
                        if type_ == 'string':
                            val = str.encode(val)
                        features_[each_key] = self.choose_write_feature_func(type_)(val)
                example = tf.train.Example(features=tf.train.Features(feature=features_))
                writer.write(example.SerializeToString())
            writer.close()

    def read_and_decode_single_example(self, file_path, num_of_epochs, columns_metadata):
        """
        Read the TFRecord data into Tensor
        :param file_path:
        :param num_of_epochs:
        :param columns_metadata:
        :return:
        """
        filenames = glob.glob(file_path)
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

    def read_data_from_tfrecord(self, file_path, metadata, batch_size=None, shuffle_batch_threads=4, capacity=10,
                                min_after_dequeue=0,
                                allow_small_final_batch=True, num_of_epochs=None):
        """
        Read the data from the TFRecords
        :param file_path:
        :param batch_size:
        :param shuffle_batch_threads:
        :param capacity:
        :param min_after_dequeue:
        :param allow_small_final_batch:
        :param num_of_epochs:
        :param metadata:
        :return:
        """
        features, key_list_ = self.read_and_decode_single_example(file_path, num_of_epochs, metadata)
        fetch_features = [features[key] for key in key_list_]
        if batch_size:
            features_batch = tf.train.shuffle_batch(
                fetch_features,
                batch_size=batch_size,
                num_threads=shuffle_batch_threads,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=allow_small_final_batch,
                name='BatchOp')
            return features_batch
        else:
            return fetch_features

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

    @staticmethod
    def change_tensor_data_to_list(data):
        """
        Function to change the type of TFRecord data from Tensor to list
        :param data:
        :return:
        """
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            data_list = sess.run(data)
            coord.request_stop()
            coord.join(threads)
            sess.close()
        return data_list
