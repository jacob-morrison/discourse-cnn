import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

checkpoint_dir = './runs/best_model/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

with tf.Session() as sess:
    model = data_helpers.load_model('./Data/GoogleNews-vectors-negative300.bin')
    print("Loading graph")
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    accuracy = tf.get_collection('accuracy')[0]
    x1 = tf.get_collection('x1')[0]
    x2 = tf.get_collection('x2')[0]
    y = tf.get_collection('y')[0]
    print("Graph loaded")
    while(1):
    	print("First sentence?")
    	sen_mat1, _ = data_helpers.get_sentence_matrix(data_helpers.pad_or_cut(raw_input().split()), model)
    	print("Second sentence?")
    	sen_mat2, _ = data_helpers.get_sentence_matrix(data_helpers.pad_or_cut(raw_input().split()), model)
    	prediction = sess.run(our_predictions, feed_dict={x1: [sen_mat1], x2: [sen_mat2], y: [0], x1_len: [1], x2_len: [1]})
    	print(prediction[0])
