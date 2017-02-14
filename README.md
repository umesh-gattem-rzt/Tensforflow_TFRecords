# Tensforflow_TFRecords

## What is TFRecords

TFRecords are the Tensorflow's default data format. A record is simply a binary file that contains serialized data which is 
tf.train.Example Protobuf object, and can be created from Python code in a few lines of code.

Below is the example to convert mnist to TFRecord.

```python
writer = tf.python_io.TFRecordWriter("mnist.tfrecords")
example = tf.train.Example(features = tf.train.Features(feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                                                   'image': tf.train.Feature(int64_list=tf.train.Int64List(value=features.astype("int64")))}))
writer.write(example.SerializeToString())
writer.close()
```

 The following are the possible types of the feature data which can be converted to TFRecord:
 
 ```python

tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))
tf.train.Feature(float_list=tf.train.FloatList(value=value_list))
tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```

## Reading data into Queues from the TFRecords:

Once your data is in format, we can make use of many tools Tensorflow provides to feed your machine learning model.

```python
def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'image': tf.FixedLenFeature([784], tf.int64)})
    label = features['label']
    image = features['image']
    return label, image
```


## tf.train.string_input_producer

### tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)

Output strings (e.g. filenames) to a queue for an input pipeline.

### Args:

1. string_tensor: A 1-D string tensor with the strings to produce.<br>
2. num_epochs: An integer (optional). If specified, string_input_producer produces each string from string_tensor num_epochs times before generating an OutOfRange error. If not specified, string_input_producer can cycle through the strings in string_tensor an unlimited number of times.
3. shuffle: Boolean. If true, the strings are randomly shuffled within each epoch.<br>
4. seed: An integer (optional). Seed used if shuffle == True.<br>
5. capacity: An integer. Sets the queue capacity.<br>
6. shared_name: (optional). If set, this queue will be shared under the given name across multiple sessions.<br>
7. name: A name for the operations (optional).<br>

### Returns:

A queue with the output strings. A QueueRunner for the Queue is added to the current Graph's QUEUE_RUNNER collection.

### Raises:

1. ValueError: If the string_tensor is a null Python list. At runtime, will fail with an assertion if string_tensor becomes a null tensor

## tf.parse_single_example

### tf.parse_single_example(serialized, features, name=None, example_names=None)

Parses a number of serialized Example protos given in serialized.

### Args

1. serialized: A scalar string Tensor, a single serialized Example. 
2. features: A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
3. name: A name for this operation (optional).
4. example_names: (Optional) A scalar string Tensor, the associated name.


## Training the Model:
 Now we have two Tensors that represents a single example. We could train with this using gradient descent. 
```python
label, image = read_and_decode_single_example("mnist.tfrecords")
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
label_val_1, image_val_1 = sess.run([label, image])

```

But it has been shown that training with batches of examples work far better than training with a single examples. 
Your estimates of gradients will have a lower variance which makes training faster. 
TensorFlow provides utilities to batch these examples and return batches.

```python
label, image = read_and_decode_single_example("mnist.tfrecords")
images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=128,capacity=2000,min_after_dequeue=1000)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
labels, images= sess.run([labels_batch, images_batch])
```

## Threading and Queues

The next major piece involves Tensorflow’s Queues and QueueRunners. 
A TensorFlow queue is quite similar to a regular queue, except all of its operations are symbolic 
and only performed on Tensorflow’s graph when needed by a sess.run.<br>
The TensorFlow Session object is multithreaded, so multiple threads can easily use the same session 
and run ops in parallel. However, it is not always easy to implement a Python program that drives threads as described above. 
All threads must be able to stop together, exceptions must be caught and reported, and queues must be properly closed when stopping.<br>
TensorFlow provides two classes to help: tf.Coordinator and tf.QueueRunner.
These two classes are designed to be used together. The Coordinator class helps multiple threads stop together and report exceptions to a program that waits for them to stop. 
The QueueRunner class is used to create a number of threads cooperating to enqueue tensors in the same queue.

## Coordinator 

The Coordinator class helps multiple threads stop together.

Its key methods are:

1. should_stop(): returns True if the threads should stop.
2. request_stop(<exception>): requests that threads should stop.
3. join(<list of threads>): waits until the specified threads have stopped.<br>
You first create a Coordinator object, and then create a number of threads that use the coordinator. 
The threads typically run loops that stop when should_stop() returns True.

Any thread can decide that the computation should stop. 
It only has to call request_stop() and the other threads will stop as should_stop() will then return True.

## QueueRunner

The QueueRunner class creates a number of threads that repeatedly run an enqueue op. 
These threads can use a coordinator to stop together. In addition, a queue runner runs a closer thread 
that automatically closes the queue if an exception is reported to the coordinator.

In the Python training program, create a QueueRunner that will run a few threads to process and enqueue examples.
Create a Coordinator and ask the queue runner to start its threads with the coordinator. 
Write a training loop that also uses the coordinator.

```python
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(100):
  '''train model'''
coord.request_stop()
coord.join(threads)
```











