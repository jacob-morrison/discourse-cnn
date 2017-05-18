# Jacob Morrison
# jacobm00
# 1242878

import sys
import pprint
import time
from random import randint

def RNA(letters):
	OPT = [[0 for i in range(len(letters))] for i in range(len(letters))]
	start_time = time.time()
	for k in xrange(4, len(letters)):
		for i in xrange(0, len(letters) - k):
			j = i + k
			# calc OPT[i, j]
			if i >= j - 4:
				OPT[i][j] = 0
			else:
				nums = []
				nums.append(OPT[i][j-1])
				for t in xrange(i, j-4):
					if (letters[t] == 'U' and letters[j] == 'A') \
						or (letters[t] == 'A' and letters[j] == 'U') \
						or (letters[t] == 'G' and letters[j] == 'C') \
						or (letters[t] == 'C' and letters[j] == 'G'):
						nums.append(OPT[i][t-1] + 1 + OPT[t+1][j-1])
				OPT[i][j] = max(nums)
	# print thing
	#for line in OPT:
	#	print(line)

	# do traceback
	trace = ['.' for i in range(len(letters))]
	start = 0
	traceback(0, len(letters), OPT, trace, letters)
	total_time = time.time() - start_time
	print(letters)
	print(''.join(trace))
	return OPT[0][len(letters)-1], total_time

def traceback(start, end, OPT, trace, letters):
	i = end - 1
	if start < end - 3:
		if OPT[start][i] == OPT[start][i-1]:
			#trace[i] = '.'
			i = i
			traceback(start, end - 1, OPT, trace, letters)
		else:
			best_t = -1
			max_score = -1
			for t in xrange(start, i-4):
				if (letters[t] == 'U' and letters[i] == 'A') \
					or (letters[t] == 'A' and letters[i] == 'U') \
					or (letters[t] == 'G' and letters[i] == 'C') \
					or (letters[t] == 'C' and letters[i] == 'G'):
					score = OPT[start][t-1] + 1 + OPT[t+1][i-1]
					if score > max_score:
						max_score = score
						best_t = t
			trace[best_t] = '('
			trace[i] = ')'
			traceback(start, best_t, OPT, trace, letters)
			traceback(best_t + 1, end - 1, OPT, trace, letters)

def generate(length):
	RNA = []
	chars = ['A', 'G', 'C', 'U']
	for i in range(length):
		RNA.append(chars[randint(0,3)])
	return ''.join(RNA)

#while(True):
#	print(str(RNA(raw_input())) + " Pairs.")
	#pairs, time = RNA(sys.argv[1])
	#print(str(pairs) + " Pairs in " + str(time) + " seconds.")
results = open('results-6.csv', 'w')
for i in xrange(20, 2000):
	string = generate(i)
	pairs, tyme = RNA(string)
	results.write(str(time) + ',' + str(i) + '\n')
	print(i)
	#print(str(pairs) + " Pairs in " + str(tyme) + " seconds.")