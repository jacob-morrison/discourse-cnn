import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

#500k iterations
checkpoint_dir = './runs/1488099198/checkpoints/'

#no dropout
#checkpoint_dir = './runs/1488069320/checkpoints/'

#with dropout
#checkpoint_dir = './runs/1488072754/checkpoints/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

with tf.Session() as sess:
	print("Loading graph")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        accuracy = tf.get_collection('accuracy')[0]
        x1 = tf.get_collection('x1')[0]
        x2 = tf.get_collection('x2')[0]
        y = tf.get_collection('y')[0]
        keep_prob = tf.get_collection('keep_prob')[0]
        print("Loaded graph")
        print("Loading model")
	sentences1, sentences2, labels = data_helpers.load_labels_and_data('./Data/GoogleNews-vectors-negative300.bin', './Data/devImplicitPDTB.txt')
	print("Loaded model")
	print("Calculating accuracy")
	accuracy = sess.run(accuracy, feed_dict={x1: sentences1, x2: sentences2, y: labels, keep_prob: 1.0})
        counts = [0 for i in range(15)]
        total = 0.
        for label in labels:
                total += 1
                counts[np.argmax(label)] += 1
        max_val = max(counts)
        print("Baseline accuracy: " + str(float(max_val) / total))
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
