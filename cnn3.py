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
batch_size = 168
display_step = 10

# network parameters
n_input = 75 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
sen_dim = 300
n_classes = 15 # 15 total senses
dropout = 0.85 # dropout probability

# tf graph input
x1 = tf.placeholder(tf.float32, [None, n_input])
x2 = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 128 inputs, 15 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([150, n_classes],dtype=tf.float32))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes],dtype=tf.float32))
}

# need to get a prediction for each sentence

# get the vector representation of each word
#pred1 = np.mean(x1, axis=1)#conv_net(x1, weights, biases, keep_prob)
#pred2 = np.mean(x2, axis=1)#conv_net(x2, weights, biases, keep_prob)

# concatenate both representations
out = tf.concat(1, [x1, x2])#[pred1, pred2])

# predict the relation class
pred = tf.add(tf.batch_matmul(out, weights['out']), biases['out'])
print(tf.shape(pred))

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
	sentences1 = np.mean(sentences1, axis=1)
        sentences2 = np.mean(sentences2, axis=1)
    # keep training until we reach max iterations
        print(len(sentences1))
	while step * batch_size < training_iters:
                start = (step * batch_size) % len(sentences1)
                end = ((step + 1) * batch_size) % len(sentences1)
                if end < start:
                        end = len(sentences1)
		batch_x1 = sentences1[start : end]
		batch_x2 = sentences2[start : end]
		batch_y = labels[start : end]
                #print(len(batch_x1))
                #print(len(batch_x2))
                #print(len(batch_y))
		sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: dropout})

		if step % display_step == 0:
			#calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.})

			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
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
        accc = sess.run(accuracy, feed_dict={x1: sentences1, x2: sentences2, y: labels, keep_prob: 1.})
        print("testing accuracy on training set: " + str(accc))
        sentences12, sentences22, labels2 = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/devImplicitPDTB.txt')
        sentences12 = np.mean(sentences12, axis=1)
        sentences22 = np.mean(sentences22, axis=1)
        accc = sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2, keep_prob: 1.})
        print("testing accuracy on dev set: " + str(accc))

