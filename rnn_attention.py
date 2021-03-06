# Jacob Morrison

import tensorflow as tf
from tensorflow.python.ops import rnn
import numpy as np
import data_helpers
import os
import time
import datetime
import sys

test = sys.argv[1]

if test == 'PDTB':
    learning_rate = 0.01
    training_iters = 100000
    n_classes = 16 # 15 total senses

elif test == 'SICK':
    learning_rate = 0.001
    training_iters = 100000
    n_classes = 3 # 15 total senses

# parameters

batch_size = 64
display_step = 10

# network parameters
n_words = 75 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
n_dim = 300
n_hidden = 300 # hidden layer num of features

# tf graph input
x1 = tf.placeholder(tf.float32, [None, n_dim, None])
x2 = tf.placeholder(tf.float32, [None, n_dim, None])
x1_len = tf.placeholder(tf.float32, [None])
x2_len = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    #'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]), dtype=tf.float32)
    'out': tf.Variable(tf.random_normal([n_dim, n_dim * n_classes],dtype=tf.float32))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, cell):
    # split input into n_words [batch_size, n_dim] tensors
    x = tf.unstack(x, n_words, 2)

    # get the output of the cell
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
    #outputs, states = rnn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=lengths, time_major=True)

    # return last output from cell
    return outputs[-1]

# this returns all outputs, which is wrong. need to find the last relevant output (aka at seqlen)
def Dynamic_RNN(x, cell, lengths):
    # get output of the cell
    outputs, _ = rnn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=lengths, time_major=False)

    return outputs

def BiRNN(x, f_cell, b_cell):

    x = tf.unstack(x, n_words, 2)

    outputs, _, _ = rnn.bidirectional_rnn(f_cell, b_cell, x, dtype=tf.float32)

    return tf.pack(tf.transpose(outputs, [1, 0, 2]))

# define an LSTM cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

# forwards cell
lstm_cell_forwards = tf.nn.rnn_cell.BasicLSTMCell(n_hidden/2, forget_bias=1.0)

# backwards cell
lstm_cell_backwards = tf.nn.rnn_cell.BasicLSTMCell(n_hidden/2, forget_bias=1.0)

# need to get a prediction for each sentence

# get the vector representation of each word
with tf.variable_scope('scope1') as scope1:
    #pred1 = Dynamic_RNN(tf.transpose(x1, perm=[0, 2, 1]), lstm_cell, x1_len)#, [-1, n_hidden])
    #pred1 = RNN(x1, lstm_cell)
    h_states1 = BiRNN(x1, lstm_cell_forwards, lstm_cell_backwards)
with tf.variable_scope('scope1') as scope1:
    scope1.reuse_variables()
    #pred2 = Dynamic_RNN(tf.transpose(x2, perm=[0, 2, 1]), lstm_cell, x2_len)#, [-1, n_hidden])
    #pred2 = RNN(x2, lstm_cell)
    h_states2 = BiRNN(x2, lstm_cell_forwards, lstm_cell_backwards)

# do something with both representations
# simple concatenation?
#out = tf.tanh(tf.concat(1, [pred1, pred2]))
x1_context = tf.reshape(tf.reduce_mean(x1, axis=2), [-1, 300, 1])
x2_context = tf.reshape(tf.reduce_mean(x2, axis=2), [-1, 300, 1])

#x1_tmp = tf.transpose(x1, [0, 2, 1])
#x2_tmp = tf.transpose(x2, [0, 2, 1])

for i in range(1):
    x1_weights = tf.nn.softmax(tf.batch_matmul(h_states1, x2_context), dim=1)
    x2_weights = tf.nn.softmax(tf.batch_matmul(h_states2, x1_context), dim=1)

    #x1_context = tf.batch_matmul(x1, x1_weights)
    #x2_context = tf.batch_matmul(x2, x2_weights)
    x1_context = tf.batch_matmul(tf.transpose(h_states1, [0, 2, 1]), x1_weights)
    x2_context = tf.batch_matmul(tf.transpose(h_states2, [0, 2, 1]), x2_weights)

x12 = tf.reshape(x1_context, [-1, n_dim])
x22 = tf.reshape(x2_context, [-1, n_dim])

# normal
'''out = tf.tanh(tf.concat(1, [x12, x22]))

# predict the relation class
pred = tf.add(tf.matmul(out, weights['out']), biases['out'])

# define loss and optimizer
'''

# bilinear
x12 = tf.tanh(x12)
x22 = tf.tanh(x22)

pred = tf.matmul(x12, weights['out'])
pred = tf.reshape(pred, [-1, n_dim, n_classes])
x22 = tf.reshape(x22,[-1, 1, n_dim])
pred = tf.batch_matmul(x22, pred)
pred = tf.reshape(pred, [-1, n_classes])
pred = tf.add(pred, biases['out'])

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
        sentences1, sentences2, labels, lengths1, lengths2 = \
            data_helpers.load_labels_and_data_PDTB(model, './Data/PDTB_implicit/train.txt', False, True, True)
    elif test == 'SICK':
        sentences1, sentences2, labels, lengths1, lengths2 = \
            data_helpers.load_data_SICK(model, './Data/SICK/train.txt', False, True, True)
    total = 0

    while total < training_iters:
        start = total  % len(sentences1)
        end = (total + batch_size) % len(sentences1)
        if end <= start:
            end = len(sentences1)
        batch_x1 = sentences1[start : end]
        batch_x2 = sentences2[start : end]
        batch_x1_lengths = lengths1[start : end]
        batch_x2_lengths = lengths2[start : end]
        batch_y = labels[start : end]
        total += (len(batch_x1))
        sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
        if step % display_step == 0:
            #calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], \
                            feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
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
        batch_x1_lengths = lengths1[start : end]
        batch_x2_lengths = lengths2[start : end]
        batch_y = labels[start : end]
        acc += (float(len(batch_x1)) / len(sentences1)) * \
                            sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, \
                                y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
        step += 1
    print(str(acc))

    # test accuracy on dev set
    print("accuracy on dev set:")
    if test == 'PDTB':
        sentences12, sentences22, labels2, lengths12, lengths22 = data_helpers.load_labels_and_data_PDTB(\
            model, \
            './Data/PDTB_implicit/dev.txt', \
            False, \
            True, True)
    elif test == 'SICK':
        sentences12, sentences22, labels2, lengths12, lengths22 = data_helpers.load_data_SICK(\
            model, \
            './Data/SICK/dev.txt', \
            False, \
            True, True)   
    print(str(sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2, x1_len: lengths12, x2_len: lengths22})))
    prediction = tf.get_collection('our_predictions')[0]
    pred = prediction.eval(feed_dict={x1: sentences12, x2: sentences22})
    results = open('results-' + test + '.txt', 'w')
    for i in range(len(labels2)):
        results.write(str(np.argmax(labels2[i], axis=0)) + "," + str(pred[i]) + "\n")
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "best_model-" + test))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_prefix = os.path.join(out_dir, "model")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path = saver.save(sess, checkpoint_prefix, global_step=1)
    print("Saved model checkpoint to {}\n".format(path))