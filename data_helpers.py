import numpy as np
import gensim as gs
import sys
from pprint import pprint

def load_labels_and_data(model_file, data_file):
	labels = {}
	print "Loading model"
	model = gs.models.Word2Vec.load_word2vec_format(model_file, binary=True)
	print "Model loaded"
	# default these to the most popular sub-categories
	labels['Temporal'] = 0
	labels['Contingency'] = 2
	labels['Comparison'] = 6
	labels['Expansion'] = 13

	# otherwise assign them as such:
	labels[('Temporal', 'Asynchronous')] = 0
	labels[('Temporal','Synchrony')] = 1
	labels[('Contingency','Cause')] = 2
	labels[('Contingency','Condition')] = 3
	labels[('Contingency','Pragmatic cause')] = 4
	labels[('Contingency','Pragmatic condition')] = 5
	labels[('Comparison','Contrast')] = 6
	labels[('Comparison','Pragmatic contrast')] = 7
	labels[('Comparison','Concession')] = 8
	labels[('Expansion','Instantiation')] = 9
	labels[('Expansion','Restatement')] = 10
	labels[('Expansion','Alternative')] = 11
	labels[('Expansion','Exception')] = 12
	labels[('Expansion','Conjunction')] = 13
	labels[('Expansion','List')] = 14

	#
	ret_labels = []
	sentences1 = []
	sentences2 = []
	with open(data_file) as f:
		for line in f:
			tokens = line.split('|')
			label = tokens[11]
			count = label.count('.')
			if count == 0:
				lab_vec = np.zeros(15)
				lab_vec[labels[label]] = 1
				ret_labels.append(lab_vec)
			else:
				strs = label.split('.')
				if strs[1] == 'Pragmatic concession':
					strs[1] = 'Concession'
				lab_vec = np.zeros(15)
				lab_vec[labels[(strs[0], strs[1])]] = 1
				#ret_labels.append(labels[(strs[0], strs[1])])
				ret_labels.append(lab_vec)
			sentence1 = pad_or_cut(tokens[24])
			sentence2 = pad_or_cut(tokens[34])
			sentences1.append(get_sentence_matrix(sentence1, model))
			sentences2.append(get_sentence_matrix(sentence2, model))

	return sentences1, sentences2, ret_labels



def pad_or_cut(sen):
	words = sen.split(" ")
	l = len(words)
	if l > 50:
		ret_sen = words[:50]
	else:
		ret_sen = words + ['<PAD>'] * (50 - l)
	return ret_sen

def get_sentence_matrix(sentence, model):
	try:
		mat = model[sentence[0]]
	except:
		mat = np.zeros(300, dtype=float)
	for i in xrange(1, len(sentence)):
		word = sentence[i]
		try:
			mat = np.column_stack([mat, model[word]])
		except:
			mat = np.column_stack([mat, np.zeros(300, dtype=float)])
	return mat


if __name__ == '__main__':
	load_labels_and_data(sys.argv[1])
