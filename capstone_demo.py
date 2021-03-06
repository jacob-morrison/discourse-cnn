import tensorflow as tf
import numpy as np
import data_helpers
import os
import time
import datetime

checkpoint_dir = './runs/best_model_bilinear-SICK/'
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
    prediction = tf.get_collection('our_predictions')[0]
    print("Graph loaded")
    while(1):
    	print("First sentence?")
    	sen_mat1, _ = data_helpers.get_sentence_matrix(data_helpers.pad_or_cut(raw_input().split()), model)
    	print("Second sentence?")
    	sen_mat2, _ = data_helpers.get_sentence_matrix(data_helpers.pad_or_cut(raw_input().split()), model)
    	pred = prediction.eval(feed_dict={x1: [sen_mat1], x2: [sen_mat2]})[0]
    	if pred == 0:
    		print("NEUTRAL\n")
    	elif pred == 1:
    		print("ENTAILMENT\n")
    	elif pred == 2:
    		print("CONTRADICTION\n")
