# Jacob Morrison

import tensorflow as tf
import numpy as np
import data_helpers
import sys
import os

test = sys.argv[1]

if test == 'PDTB':
	learning_rate = 0.01
	training_iters = 1000000
	n_classes = 16 # 16 total senses

elif test == 'SICK':
	learning_rate = 0.01
	training_iters = 1000000
	n_classes = 3

batch_size = 64
display_step = 10

# network parameters
n_input = 75 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
sen_dim = 300

# tf graph input
x1 = tf.placeholder(tf.float32, [None, sen_dim, n_input])
x2 = tf.placeholder(tf.float32, [None, sen_dim, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Store layers weight & bias
weights = {
	'w': tf.constant(1.0/75, dtype=tf.float32, shape=[n_input,1]),
	#'w': tf.Variable(tf.random_normal([n_input,1], mean=1.0/75, stddev=1/300, dtype=tf.float32)),
	#'w': tf.Variable(tf.random_normal([n_input,1], mean=1.0/75, stddev=0, dtype=tf.float32)),
	'out': tf.Variable(tf.random_normal([sen_dim, sen_dim * n_classes],dtype=tf.float32))
}

biases = {
	'out': tf.Variable(tf.random_normal([n_classes],dtype=tf.float32)),
}

x12 = tf.reshape(x1, [-1, n_input])                                                                                                                                                 
x22 = tf.reshape(x2, [-1, n_input])                                                                                                                                                 
x12 = tf.matmul(x12, weights['w'])                                                                                                                                            
x22 = tf.matmul(x22, weights['w'])
x12 = tf.reshape(x12, [-1, sen_dim])
x22 = tf.reshape(x22, [-1, sen_dim])
x12 = tf.tanh(x12)
x22 = tf.tanh(x22)

pred = tf.matmul(x12, weights['out'])
pred = tf.reshape(pred, [-1, sen_dim, n_classes])
x22 = tf.reshape(x22,[-1, 1, sen_dim])
pred = tf.batch_matmul(x22, pred)
pred = tf.reshape(pred, [-1, n_classes])
pred = tf.add(pred, biases['out'])

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
our_predictions = tf.argmax(pred, 1)
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
tf.add_to_collection('our_predictions', our_predictions)
with tf.Session() as sess:
	sess.run(init)
	step = 1
	model = data_helpers.load_model('./Data/GoogleNews-vectors-negative300.bin')
	if test == 'PDTB':
		sentences1, sentences2, labels = data_helpers.load_labels_and_data_PDTB(model, './Data/PDTB_implicit/train.txt')
	elif test == 'SICK':
		sentences1, sentences2, labels = data_helpers.load_data_SICK(model, './Data/SICK/train.txt')
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
			# extract variables here
			#w2 = sess.run(weights['w2'])
			#print(w2)
		step += 1
	print "Training finished!"

	# calculate training set accuracy
	print("testing accuracy on training set: ")
	step = 0
	acc = 0.
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

	# test accuracy on dev set
	print("accuracy on dev set:")
	if test == 'PDTB':
		sentences12, sentences22, labels2 = data_helpers.load_labels_and_data_PDTB(\
			model, \
			'./Data/PDTB_implicit/dev.txt')
	elif test == 'SICK':
		sentences12, sentences22, labels2 = data_helpers.load_data_SICK(\
			model, \
			'./Data/SICK/dev.txt')
	print(str(sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2})))
	prediction = tf.get_collection('our_predictions')[0]
	pred = prediction.eval(feed_dict={x1: sentences12, x2: sentences22})
	results = open('bilinear-results-' + test + '.txt', 'w')
	for i in range(len(labels2)):
		results.write(str(np.argmax(labels2[i], axis=0)) + "," + str(pred[i]) + "\n")
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "best_model_bilinear-" + test))
	# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	checkpoint_prefix = os.path.join(out_dir, "model")
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	path = saver.save(sess, checkpoint_prefix, global_step=1)
	print("Saved model checkpoint to {}\n".format(path))

