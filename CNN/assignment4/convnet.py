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

######################################################################################
image_size = 28
num_labels = 10
num_channels = 1

batch_size = 128
patch_size = 5
depth = 16
num_hidden = 64

num_steps = 8001
##############################################################################

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
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


graph = tf.Graph()

with graph.as_default():

	#Input
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	keep_prob = tf.placeholder(tf.float32)

	#Parameters
	conv1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
	conv1_biases = tf.Variable(tf.zeros([depth]))

	conv2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	conv2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

	fc1_weights = tf.Variable(tf.truncated_normal([(image_size // 4) * (image_size // 4) * depth, num_hidden], stddev=0.1))
	fc1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

	fc2_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	fc2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	global_step = tf.Variable(0, trainable=False)
  	starter_learning_rate = 0.06

	#Model
	def model(data):
		conv = tf.nn.conv2d(data, conv1_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.dropout(tf.nn.max_pool(tf.nn.relu(conv + conv1_biases), [1,2,2,1], [1,2,2,1], padding = 'SAME'), keep_prob)
		conv = tf.nn.conv2d(hidden, conv2_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.dropout(tf.nn.max_pool(tf.nn.relu(conv + conv2_biases), [1,2,2,1], [1,2,2,1], padding = 'SAME'), keep_prob)
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases), keep_prob)
		out = tf.matmul(hidden, fc2_weights) + fc2_biases
		return out

	#Training
	logits = model(tf_train_dataset) 
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + 0.005*tf.nn.l2_loss(conv1_weights) + 0.005*tf.nn.l2_loss(conv2_weights) + 0.005*tf.nn.l2_loss(fc1_weights) + 0.005*tf.nn.l2_loss(fc2_weights))
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)

	#Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	#Predictions
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))



with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	for step in range(num_steps):
		offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.6}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 1000 == 0):
		  print('Minibatch loss at step %d: %f' % (step, l))
		  print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
		  print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval({keep_prob:1.0}), valid_labels))
  	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval({keep_prob:1.0}), test_labels))


