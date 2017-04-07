"""
| **@created on:** 20/05/16,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| File Utilities
|
| Sphinx Documentation Status:** Complete
|
..todo::
"""

from ast import literal_eval
from typeguard import typechecked
import numpy as np
import pandas as pd
from pyhdfs import HdfsClient
from typing import Union
import csv
import os
import glob
import tensorflow as tf
from collections import OrderedDict


class TFRecords(object):
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

    def convert_and_read_data_from_tfrecords(self, filepath, delimiter, metadata, output_label, tfrecords_path,
                                             batch_size, num_of_epochs, header=None, label_vector=False,
                                             index_col=False,
                                             shuffle_batch_threads=1, capacity=10, min_after_deque=0,
                                             allow_small_final_batch=True):
        """
        | **@author:**  Umesh Kumar
        |
        | Read the raw data from the file_path given and return the data needed to create tfrecords
        :param filepath: file path to read raw data
        :param delimiter: delimiter for column values
        :param metadata: The dict which contains the input and outputs keys with no of values.
        :param output_label: Its a index from where the output label starts.
        :param tfrecords_path: Path where TFRecord should be stored/
        :param batch_size: Batch size for training data.
        :param header: This specifies whether header is present in CSV or not
        :param index_col: The first column will be act as index_col if is true
        :param num_of_epochs: No of epochs for training.
        :param label_vector: True if output_label is vector.
        :param shuffle_batch_threads: no of threads for queue.
        :param capacity: Capacity of the queue.
        :param min_after_deque:
        :param allow_small_final_batch: It returns small batch remained after batching if specified as True.
        """
        data_set = []
        total_samples = 0
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader) if header else reader
            for line in reader:
                id = line[1] if index_col else total_samples
                total_samples += 1
                line = line[1:] if index_col else line
                if label_vector:
                    output = literal_eval(line[-1])
                    input = [float(val) for val in line[:-1]]
                    line_data = Data(str(id), input, output)
                else:
                    line = [float(val) for val in line]
                    line_data = Data(str(id), line[:output_label], line[output_label:])
                data_set.append(line_data)
        record_file_path = self.convert_to_tfrecord(tfrecords_directory=tfrecords_path,
                                                    record_name=os.path.split(filepath)[1].replace('.csv', ''),
                                                    metadata=metadata, data_set=data_set)
        return self.setup_data_read(tfrecords_path=record_file_path, batch_size=batch_size, num_of_epochs=num_of_epochs,
                                    shuffle_batch_threads=shuffle_batch_threads,
                                    capacity=capacity,
                                    min_after_deque=min_after_deque, allow_small_final_batch=allow_small_final_batch,
                                    metadata=metadata)

    def convert_to_tfrecord(self, tfrecords_directory, record_name, metadata, data_set):
        """
        Converts the dataset to tfrecords. Writes the records into specified directory path
        :param tfrecords_directory: directory to save tfrecords
        :param record_name: name of the tfrecord
        :param metadata: The dict which contains the input and outputs keys with no of values.
        :param data_set: type > list of objects, this is the dataset to be converted into tfrecords
        :return:
        """
        record_filepath = os.path.join(tfrecords_directory, record_name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(record_filepath)
        for data_ in data_set:
            features_ = OrderedDict()
            for type_ in metadata:
                for each_key in metadata[type_]:
                    val = getattr(data_, each_key)
                    if type_ == 'string':
                        val = str.encode(val)
                    features_[each_key] = self.choose_write_feature_func(type_)(val)
            example = tf.train.Example(features=tf.train.Features(feature=features_))
            writer.write(example.SerializeToString())
        writer.close()
        return record_filepath

    def read_and_decode_single_example(self, file_path, num_of_epochs, columns_metadata):
        """
        Read and decode the tfrecord
        :param file_path: Path where TFRecords is stored
        :param num_of_epochs: Num of epochs
        :param columns_metadata: The dict which contains the input and outputs keys with no of values.
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

    def setup_data_read(self, tfrecords_path, batch_size, num_of_epochs, metadata, shuffle_batch_threads=1, capacity=10,
                        min_after_deque=0, allow_small_final_batch=True):
        """
        Setup the data read controller
        :param tfrecords_path: directory path where list of tfrecords are saved
        :param metadata: The dict which contains the input and outputs keys with no of values.
        :param batch_size: samples in each batch
        :param shuffle_batch_threads: number of threads to use
        :param capacity: capacity of the queue
        :param min_after_deque: minimum samples after deque
        :param allow_small_final_batch: allow_small_final_batch
        :param num_of_epochs: num of epochs of data needed
        :return:
        """
        features, key_list_ = self.read_and_decode_single_example(tfrecords_path, num_of_epochs, metadata)
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

    @staticmethod
    def next_batch(session, features_batch, metadata):
        """
        Call this function repeatedly to fetch all the data. (Number of calls depends on the batch size and total data size)
        :param session: TF session
        :param features_batch: list of features which can be read batchwise
        :param metadata: The dict which contains the input and outputs keys with no of values.
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


def convert_df_tolist(*input_data):
    """
    | **@author:** Prathyush SP
    |
    | Convert Dataframe to List
    |
    :param input_data: Input Data (*args)
    :return: Dataframe
    .. todo::
        Prathyush SP:
            1. Check list logic
            2. Perform Dataframe Validation
    """
    dataframes = []
    for df in input_data:
        if len(df) == 0:
            dataframes.append([])
        else:
            if isinstance(df, pd.DataFrame):
                if len(input_data) == 1:
                    return df.values.tolist()
                dataframes.append(df.values.tolist())
            elif isinstance(df, pd.Series):
                df_list = df.to_frame().values.tolist()
                if isinstance(df_list, list):
                    if isinstance(df_list[0][0], list):
                        dataframes.append([i[0] for i in df.to_frame().values.tolist()])
                    else:
                        dataframes.append(df.to_frame().values.tolist())
                else:
                    dataframes.append(df.to_frame().values.tolist())
    return dataframes


@typechecked()
def read_csv(filename: str = None, data: Union[list, np.ndarray] = None, split_ratio: Union[list, None] = (50, 20, 30),
             delimiter: str = ',',
             normalize: bool = False, dtype=None,
             header: bool = None, skiprows: int = None,
             index_col: int = False, output_label=True, randomize: bool = False, return_as_dataframe: bool = False,
             describe: bool = False,
             label_vector: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | The function is used to read a csv file with a specified delimiter
    :param filename: File name with absolute path
    :param data: Data used for train and test
    :param split_ratio: Ratio used to split data into train and test
    :param delimiter: Delimiter used to split columns
    :param normalize: Normalize the Data
    :param dtype: Data Format
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param index_col: Index Column
    :param output_label: Column which specifies whether output label should be available or not.
    :param randomize: Randomize data
    :param return_as_dataframe: Returns as a dataframes
    :param describe: Describe Input Data
    :param label_vector: True if output label is a vector
    :return: return train_data, train_label, test_data, test_label based on return_as_dataframe
    """
    if filename:
        df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype, skiprows=skiprows)
    elif data.any():
        df = pd.DataFrame(data)
    else:
        raise Exception('Filename / Data are None. Specify atleast one source')
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    df = df.apply(lambda x: np.log(x)) if normalize else df
    if not split_ratio:
        if output_label is None or output_label is False:
            if return_as_dataframe:
                return df
            else:
                return convert_df_tolist(df)
        else:
            column_drop = len(df.columns) - 1 if output_label is True else output_label
            label = df[column_drop].apply(literal_eval) if label_vector else df[column_drop]
            data = df.drop(column_drop, axis=1)
            if return_as_dataframe:
                return data, label
            else:
                return convert_df_tolist(data, label)
    if len(split_ratio) == 3 and sum(split_ratio) == 100:
        test_size = int(len(df) * split_ratio[-1] / 100)
        valid_size = int(len(df) * split_ratio[1] / 100)
        train_size = int(len(df) - (test_size + valid_size))
        train_data_df, valid_data_df, test_data_df = df.head(int(train_size)), \
                                                     df.iloc[train_size:(train_size + valid_size)], \
                                                     df.tail(int(test_size))
        if output_label is None or output_label is False:
            if return_as_dataframe:
                return train_data_df, valid_data_df, test_data_df
            else:
                return convert_df_tolist(train_data_df, valid_data_df, test_data_df)
        elif output_label is not None:
            column_drop = len(train_data_df.columns) - 1 if output_label is True else output_label
            train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[
                column_drop]
            train_data_df = train_data_df.drop(column_drop, axis=1)
            valid_label_df = valid_data_df[column_drop].apply(literal_eval) if label_vector else valid_data_df[
                column_drop]
            valid_data_df = valid_data_df.drop(column_drop, axis=1)
            test_label_df = test_data_df[column_drop].apply(literal_eval) if label_vector else test_data_df[
                column_drop]
            test_data_df = test_data_df.drop(column_drop, axis=1)
            if return_as_dataframe:
                return train_data_df, train_label_df, valid_data_df, valid_label_df, test_data_df, test_label_df
            else:
                return convert_df_tolist(train_data_df, train_label_df, valid_data_df,
                                         valid_label_df, test_data_df, test_label_df)
    else:
        raise Exception("Length of split_ratio should be 2 or 3 with sum of elements equal to 100")


def read_hdfs(filename, host, split_ratio, delimiter=',', normalize=False, dtype=None, header=None, skiprows=None,
              index_col=False, output_label=True, randomize=False, return_as_dataframe=False, describe=False,
              label_vector=False):
    client = HdfsClient(hosts=host)
    return read_csv(client.open(filename), split_ratio, delimiter=delimiter, normalize=normalize, dtype=dtype,
                    header=header, skiprows=skiprows, index_col=index_col, output_label=output_label,
                    randomize=randomize, return_as_dataframe=return_as_dataframe, describe=describe,
                    label_vector=label_vector)


class Data:
    """
    Data object for creating the tfrecords
    """

    def __init__(self, id, input, output):
        self.id = id
        self.input = input
        self.output = output
