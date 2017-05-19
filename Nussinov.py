import sys
import time

def findPairs(sequence):
    OPT = [[0 for  i in range(len(sequence))] for i in range(len(sequence))]
    start = time.time()
    for k in xrange(4, len(sequence)):
        for i in xrange(0, len(sequence) - k):
            j = i + k
            if i >= j - 4:
                OPT[i][j] = 0
            else:
                findMax = []
                for t in xrange(i, j - 4):
                    letterOne = sequence[j]
                    letterTwo = sequence[t]
                    if canMatch(letterOne, letterTwo):
                        findMax.append(OPT[i][t - 1] + OPT[t+1][j - 1] + 1)
                if len(findMax) > 0:
                    OPT[i][j] = max(max(findMax), OPT[i][j - 1])
                else:
                    OPT[i][j] = OPT[i][j - 1]
    result = ['.' for i in range(len(sequence))]
    traceBack(sequence, 0, len(sequence) - 1, result, OPT)
    end = time.time()
    print(''.join(result))
    print(str(OPT[0][len(sequence) - 1]) + " Pairs.")
    print(str(end - start) + " seconds.")
    if len(sequence) < 26:
        for line in OPT:
            print(line)
    return end - start

def traceBack(sequence, start, end, result, OPT):
    if start < end - 3:
        if OPT[start][end] == OPT[start][end - 1]:
            traceBack(sequence, start, end - 1, result, OPT)
        else:
            max = 0
            index = 0
            for t in xrange(start, end - 4):
                if canMatch(sequence[end], sequence[t]):
                    if 1 + OPT[t+1][end-1] + OPT[start][t-1] > max:
                        max = 1 + OPT[t+1][end-1] + OPT[start][t-1]
                        index = t
            result[end] = ')'
            result[index] = '('
            traceBack(sequence, index + 1, end - 1, result, OPT)
            traceBack(sequence, start, index, result, OPT)

def canMatch(letterOne, letterTwo):
    return (letterOne == 'A' and letterTwo == 'U') or (letterOne == 'U' and letterTwo == 'A') or (letterOne == 'C' and letterTwo == 'G') or (letterOne == 'G' and letterTwo == 'C')

def makeString(size):
    letters = ['G', 'C', 'U', 'A']
    string = []
    for k in range(size):
        string.append(letters[k % 4])
    return string

results = open("results.csv", 'w')
for i in xrange(20, 300):
    sequence = makeString(i)
    seconds = findPairs(sequence)
    results.write(str(i) + ',' + str(seconds) + '\n')
    print(i)
for i in xrange(300, 1000, 100):
    sequence = makeString(i)
    seconds = findPairs(sequence)
    results.write(str(i) + ',' + str(seconds) + '\n')
    print(i)
for i in xrange(1000, 2001, 250):
    sequence = makeString(i)
    seconds = findPairs(sequence)
    results.write(str(i) + ',' + str(seconds) + '\n')
    print(i)

#print(sys.argv[1])
#findPairs(sys.argv[1])