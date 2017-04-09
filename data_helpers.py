import numpy as np
import gensim as gs
import sys
from collections import Counter
from pprint import pprint

def load_model(model_file):
	print "Loading model"
	model = gs.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
	print "Model loaded"
	return model

def load_labels_and_sentences(data_file):
	labels	# otherwise assign them as such:
	labels[('Temporal', 'Asynchronous')] = 0
	labels[('Temporal','Synchrony')] = 1
	labels[('Contingency','Cause')] = 2
	labels[('Contingency','Condition')] = 3
	labels[('Contingency','Pragmatic cause')] = 4
	labels[('Contingency','Pragmatic condition')] = 5
	labels[('Comparison','Contrast')] = 6
	labels[('Comparison','Pragmatic contrast')] = 7
	labels[('Comparison','Concession')] = 8
	labels[('Comparison','Pragmatic concession')] = 9
	labels[('Expansion','Instantiation')] = 10
	labels[('Expansion','Restatement')] = 11
	labels[('Expansion','Alternative')] = 12
	labels[('Expansion','Exception')] = 13
	labels[('Expansion','Conjunction')] = 14
	labels[('Expansion','List')] = 15

	ret_labels = []
	sentences1 = []
	sentences2 = []
	with open(data_file) as f:
		for line in f:
			tokens = line.split('|')
			label = tokens[11]
			count = label.count('.')
			if count > 0:
				strs = label.split('.')
				lab_vec = np.zeros(16)
				lab_vec[labels[(strs[0], strs[1])]] = 1
				ret_labels.append(lab_vec)
				sentences1.append(tokens[24])
				sentences2.append(tokens[34])
		total = len(ret_labels)
		ls = Counter()
		for l in ret_labels:
			idx = np.argmax(l)
			ls[idx] += 1
		mx = 0
		for i in range(15):
			if ls[i] > mx:
				mx = ls[i]
		print("Accuracy if we did nothing: " + str(float(mx)/total))
	return sentences1, sentences2, ret_labels

def load_labels_and_data(model, data_file, smallSentences=False):
	labels = {}
	#print "Loading model"
	#model = gs.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
	#model = gs.models.Word2Vec.load_word2vec_format(model_file, binary=True)
	#print "Model loaded"
	# default these to the most popular sub-categories
	#labels['Temporal'] = 0
	#labels['Contingency'] = 2
	#labels['Comparison'] = 6
	#labels['Expansion'] = 13

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
	labels[('Comparison','Pragmatic concession')] = 9
	labels[('Expansion','Instantiation')] = 10
	labels[('Expansion','Restatement')] = 11
	labels[('Expansion','Alternative')] = 12
	labels[('Expansion','Exception')] = 13
	labels[('Expansion','Conjunction')] = 14
	labels[('Expansion','List')] = 15

	#
	ret_labels = []
	sentences1 = []
	sentences2 = []
	with open(data_file) as f:
		for line in f:
			tokens = line.split('|')
			label = tokens[11]
			count = label.count('.')
			if count > 0:
				strs = label.split('.')
				# ?? shouldn't be doing this anymore
				#if strs[1] == 'Pragmatic concession':
				#	strs[1] = 'Concession'
				lab_vec = np.zeros(16)
				lab_vec[labels[(strs[0], strs[1])]] = 1
				#ret_labels.append(labels[(strs[0], strs[1])])
				ret_labels.append(lab_vec)
				sentence1 = pad_or_cut(tokens[24])
				sentence2 = pad_or_cut(tokens[34])
				bigSen1, smallSen1 = get_sentence_matrix(sentence1, model)
				bigSen2, smallSen2 = get_sentence_matrix(sentence2, model)
				if (smallSentences):
					sentences1.append(smallSen1)
					sentences2.append(smallSen2)
				else:
					sentences1.append(bigSen1)
					sentences2.append(bigSen2)
		total = len(ret_labels)
		ls = Counter()
		for l in ret_labels:
			idx = np.argmax(l)
			ls[idx] += 1
		mx = 0
		for i in range(15):
			if ls[i] > mx:
				mx = ls[i]
		print("Accuracy if we did nothing: " + str(float(mx)/total))
	return sentences1, sentences2, ret_labels

def test(data_file):
	total = 0
	total50 = 0
	total75 = 0
	total100 = 0
	with open(data_file) as f:
		for line in f:
			total += 2
			tokens = line.split("|")
			sen1 = tokens[24].split()
			sen2 = tokens[34].split()
			if len(sen1) > 50:
				total50 += 1
			if len(sen1) > 75:
				total75 += 1
			if len(sen1) > 100:
				total100 += 1
			if len(sen2) > 50:
				total50 += 1
			if len(sen2) > 75:
				total75 += 1
			if len(sen2) > 100:
				total100 += 1
	print(str(total) + " " + str(total50) + " " + str(total75) + " " + str(total100))

def pad_or_cut(sen):
	sen_len = 75
	words = sen.replace('\'', ' \' ').replace('"', ' " ').replace('.', ' . ').replace(',', ' , ').replace('-', ' - ').split(" ")
	print(words)
	l = len(words)
	if l > sen_len:
		ret_sen = words[:sen_len]
	else:
		ret_sen = words + ['<PAD>'] * (sen_len - l)
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
	return mat, np.mean(mat, axis=1)


if __name__ == '__main__':
	test(sys.argv[1])
