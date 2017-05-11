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
def accuracy(predictions, trueLabels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(trueLabels, 1)) / predictions.shape[0])

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


def accuracy(predictions, trueLabels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(trueLabels, 1)) / predictions.shape[0])

batch_size = 128
num_steps = 10001
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():
  #Input
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  keep_prob = tf.placeholder(tf.float32)

  #parameters
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes])) 
  biases = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels])) 
  biases2 = tf.Variable(tf.zeros([num_labels]))
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.5
  

  #training
  logits = tf.nn.relu(tf.matmul(tf_train_dataset, weights) + biases)
  logits_drop = tf.nn.dropout(logits, keep_prob)
  logits2 = tf.matmul(logits_drop, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2) + 0.01*tf.nn.l2_loss(weights) + 0.01*tf.nn.l2_loss(weights2))
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)

  #optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  #predictions
  train_prediction = tf.nn.softmax(logits2)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), keep_prob), weights2) + biases2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), keep_prob), weights2) + biases2)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized!!!')
  start = time.time()
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if step%500==0:
      #print("Minibatch loss at step %d: %f" % (step, l))
      #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval({keep_prob:1.0}), valid_labels))
  end = time.time()
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval({keep_prob:1.0}), test_labels))
  print("Time elapsed: %.2f" % (end-start))

