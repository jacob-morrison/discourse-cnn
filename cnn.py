# Jacob Morrison

import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

# parameters
learning_rate = 0.01
training_iters = 300000
batch_size = 128
display_step = 10

# network parameters
n_input = 50 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
sen_dim = 300
n_classes = 16 # 16 total senses
dropout = 0.60 # dropout probability

# tf graph input
x1 = tf.placeholder(tf.float32, [None, sen_dim, n_input])
x2 = tf.placeholder(tf.float32, [None, sen_dim, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# create some wrappers (we're following a guide, woo)
def conv2d(x, W, b, strides=1):
	# conv2d wrapper with bias and relu (?) activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv1d(x, W, b, strides=1):
        x = tf.nn.conv1d(x, W, strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

def maxpool1d(x, k=2):
        # MaxPool1D wrapper
        return tf.nn.max_pool(tf.reshape(x, shape=[-1, 1, sen_dim * n_input, 1]), ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# create model
def conv_net(x, weights, biases, dropout):
	# reshape input
	#x = tf.reshape(x, shape=[-1, sen_dim * n_input, 1])
	x = tf.reshape(x, shape=[-1, sen_dim, n_input, 1])
        # convolutional layer
	#conv1 = conv1d(x, weights['wc1'], biases['bc1'])
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        #max pooling (down-sampling)
	#conv1 = maxpool1d(conv1, k=2)
        conv1 = maxpool1d(conv1, k=2)
        #return tf.reshape(conv1, shape=[-1, 300*25*32])
        return tf.reshape(conv1, shape=[-1, 998343])
        # Convolution Layer
	#conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        #Max Pooling (down-sampling)
	#conv2 = maxpool2d(conv2, k=2)
        #return tf.reshape(conv2, [-1, 300*16*25])
    # fully connected layer
    # reshape conv2 output to fit the fully connected layer input
        #fc1 = tf.reshape(conv2, [-1, 300*16*7])
	#fc1 = tf.reshape(conv1, [-1, 300*16*25])
        #fc1 = tf.nn.bias_add(tf.batch_matmul(fc1, weights['wd1']), biases['bd1'])
	#fc1 = tf.nn.relu(fc1)
	# apply dropout
	#fc1 = tf.nn.dropout(fc1, dropout)

	return fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 5, 1, 32],dtype=tf.float32)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 1, 32, 64],dtype=tf.float32)),
    # fully connected, 7*7*64 inputs, 64 outputs
        'wd1': tf.Variable(tf.random_normal([998343, 256],dtype=tf.float32)),
    # 128 inputs, 15 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([998343*2, n_classes],dtype=tf.float32))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32],dtype=tf.float32)),
    'bc2': tf.Variable(tf.random_normal([64],dtype=tf.float32)),
        'bd1': tf.Variable(tf.random_normal([256],dtype=tf.float32)),
    'out': tf.Variable(tf.random_normal([n_classes],dtype=tf.float32))
}

# need to get a prediction for each sentence

# get the vector representation of each word
print(tf.shape(x1))
pred1 = conv_net(x1, weights, biases, keep_prob)
pred2 = conv_net(x2, weights, biases, keep_prob)

# concatenate both representations
out = tf.concat(1, [pred1, pred2])

# predict the relation class
pred = tf.add(tf.batch_matmul(out, weights['out']), biases['out'])

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing all variables
init = tf.global_variables_initializer()

# launch the graph
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
tf.add_to_collection('accuracy', accuracy)
tf.add_to_collection('x1', x1)
tf.add_to_collection('x2', x2)
tf.add_to_collection('y', y)
tf.add_to_collection('keep_prob', keep_prob)
with tf.Session() as sess:
	sess.run(init)
	step = 1
	sentences1, sentences2, labels = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/implicitTrainPDTB.txt')
	# keep training until we reach max iterations
        #print(len(sentences1))
        total = 0
	while total < training_iters:
                start = total  % len(sentences1)
                end = (total + batch_size) % len(sentences1)
                if end <= start:
                        end = len(sentences1)
		batch_x1 = sentences1[start : end]
		batch_x2 = sentences2[start : end]
		batch_y = labels[start : end]
                total += (len(batch_x1))
		sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: dropout})

		if step % display_step == 0:
			#calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.})

			print "Iter " + str(total) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)

		step += 1

	print "Training finished!"

	#then save model
        timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	path = saver.save(sess, checkpoint_prefix, global_step=1)
	print("Saved model checkpoint to {}\n".format(path))

        print("testing accuracy on training set: ")#
        step = 0
        acc = 0.
        #sentences1, sentences2, labels = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/implicitTrainPDTB.txt')
        print(len(sentences1))
        batch_size2 = batch_size * 2
        while step * batch_size2 < len(sentences1):
                start = (step * batch_size2)
                end = ((step + 1) * batch_size2)
                if end > len(sentences1):
                        end = len(sentences1)
                batch_x1 = sentences1[start : end]
                batch_x2 = sentences2[start : end]
                batch_y = labels[start : end]
                acc += (float(len(batch_x1)) / len(sentences1)) * sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.})
                step += 1
        print(str(acc))
        #print(str(sess.run(accuracy, feed_dict={x1: sentences1, x2: sentences2, y: labels, keep_prob: 1.})))
        sentences12, sentences22, labels2 = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/devImplicitPDTB.txt')                          

        print(len(sentences12))
        batch_size2 = batch_size * 2
        step = 0
        acc = 0.
        while step * batch_size2 < len(sentences12):
                start = (step * batch_size2)
                end = ((step + 1) * batch_size2)
                if end > len(sentences12):
                        end = len(sentences12)
                if start >= len(sentences12):
                        break;
                batch_x1 = sentences12[start : end]
                batch_x2 = sentences22[start : end]
                batch_y = labels2[start : end]
                acc += (float(len(batch_x1)) / len(sentences12)) * sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.})
                step += 1
        #print(str(acc/(len(sentences1)/batch_size)))
        print(str(acc))
        
        print(str(sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2, keep_prob: 1.})))
