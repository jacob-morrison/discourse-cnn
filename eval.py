import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

checkpoint_dir = '.runs/1488059216/checkpoints/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

with tf.Session() as sess:
	print("Loading graph")
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    print("Loaded graph")
    print("Loading model")
	sentences1, sentences2, labels = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/devImplicitPDTB.txt')
	print("Loaded model")
	print("Calculating accuracy")
	accuracy = sess.run(accuracy, feed_dict={x1: sentences1, x2: sentences2, y: labels, keep_prob: 1.0})
	print("Accuracy: " + "{:.5f}".format(accuracy))


	'''
	while step * batch_size < training_iters:
                start = (step * batch_size) % len(sentences1)
                end = ((step + 1) * batch_size) % len(sentences1)
                if end < start:
                        end = len(sentences1)
		batch_x1 = sentences1[start : end]
		batch_x2 = sentences2[start : end]
		batch_y = labels[start : end]
		sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: dropout})

		if step % display_step == 0:
			#calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.})

			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Evaluation Accuracy= " + \
                  "{:.5f}".format(acc)

		step += 1
	'''

	print "Evaluation finished!"