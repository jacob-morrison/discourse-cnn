# Jacob Morrison

import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

# parameters
learning_rate = 0.01
training_iters = 200000
batch_size = 128
display_step = 10

# network parameters
n_input = 75 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
n_steps = 300
n_hidden = 128 # hidden layer num of features
n_classes = 16 # 15 total senses

# tf graph input
x1 = tf.placeholder(tf.float32, [None, n_steps, n_input])
x2 = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x1, x2):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x1 = tf.transpose(x1, [1, 0, 2])
    x2 = tf.transpose(x2, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x1 = tf.reshape(x1, [-1, n_steps])
    x2 = tf.reshape(x2, [-1, n_steps])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x1 = tf.split(0, n_input, x1)
    x2 = tf.split(0, n_input, x2)

    # Define a lstm cell with tensorflow
    with tf.variable_scope('lstm1'):
        lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs1, states1 = tf.nn.rnn(lstm_cell1, x1, dtype=tf.float32)
    with tf.variable_scope('lstm2'):
        lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs2, states2 = tf.nn.rnn(lstm_cell2, x2, dtype=tf.float32)

    return tf.concat(1, [outputs1[-1], outputs2[-1]])

# need to get a prediction for each sentence

# get the vector representation of each word
#pred1 = RNN(x1)
#pred2 = pred1
#pred2 = RNN(x2)

# concatenate both representations
#out = tf.concat(1, [pred1, pred2])
out = RNN(x1, x2)

# predict the relation class
pred = tf.add(tf.batch_matmul(out, weights['out']), biases['out'])

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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
		sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
		if step % display_step == 0:
			#calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
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
                acc += (float(len(batch_x1)) / len(sentences1)) * sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
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
                acc += (float(len(batch_x1)) / len(sentences12)) * sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
                step += 1
        #print(str(acc/(len(sentences1)/batch_size)))
        print(str(acc))
        
        print(str(sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2})))
