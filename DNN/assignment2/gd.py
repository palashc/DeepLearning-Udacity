from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pickle_file = '../../ML2DL/assignment-1/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  # print('Training set', train_dataset.shape, train_labels.shape)
  # print('Validation set', valid_dataset.shape, valid_labels.shape)
  # print('Test set', test_dataset.shape, test_labels.shape)



image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)


train_subset_array = [5000, 10000, 15000, 20000, 25000, 30000]
num_steps_array = [5000]
output_file = "out.txt"
f = open(output_file, "w")

def accuracy(predictions, trueLabels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(trueLabels, 1)) / predictions.shape[0])

for train_subset in train_subset_array:
  for num_steps in num_steps_array:
    graph = tf.Graph()
    with graph.as_default():
      #Input
      tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
      tf_train_labels = tf.constant(train_labels[:train_subset])
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      #parameters
      weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels])) 
      biases = tf.Variable(tf.zeros([num_labels]))

      #training
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      #optimizer
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      #predictions
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    start = time.time()

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      #print('Initialized!!!')
      for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if step%500==0:
          end = time.time()
          print('SubsetSize: %d NumSteps: %d TimeElapsed: %.2f Test accuracy: %.1f%%' % (train_subset, step, (end-start), accuracy(test_prediction.eval(), test_labels)))
          f.write('SubsetSize: %d NumSteps: %d TimeElapsed: %.2f Test accuracy: %.1f%%' % (train_subset, step, (end-start), accuracy(test_prediction.eval(), test_labels)))
      
      
f.close()