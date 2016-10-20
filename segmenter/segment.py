# coding=utf-8

import sys, codecs, optparse, os
import heapq, math, itertools
from scipy import linalg
import scipy.interpolate as interp
from numpy import c_, log, exp, sqrt, mean, median
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w',
                     default=os.path.join('data', 'count_1w.txt'),
                     help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w',
                     default=os.path.join('data', 'count_2w.txt'),
                     help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input",
                     default=os.path.join('data', 'input'),
                     help="input file to segment")
optparser.add_option("-a", "--alpha", dest="alpha",
                     help="parameter alpha")
optparser.add_option("-g", "--gamma", dest="gamma",
                     help="parameter gamma")
# EVERY BIGRAM PROBABILITY IS MULTIPLIED BY THIS CONSTANT
optparser.add_option("-k", "--kappa", dest="kappa",
                     help="parameter kappa")

# IF USING CORPUS ADAPTATION, THIS IS THE PROPORTION OF TOP
# SENTENCES THAT GET FED BACK INTO THE CORPUS
optparser.add_option("-l", "--beta", dest="beta",
                     help="parameter beta")

(opts, _) = optparser.parse_args()

# just configure the path to the count files here
unigramPath = "./data/count_1w.txt"
bigramPath = "./data/count_2w.txt"


class Corpus(dict):
    '''
    This class stores a hierarchical dictionary, each entry is a list that
    represents a word, w. The first entry of the list is the unigram
    probability of w occurring, and the second entry is a
    dictionary whose keys are words that come AFTER w in the bigram
    data. The values are the conditional probabilities of w occurring before
    the other word.

    TODO: figure out what to do about missing words

    Convoluted? Maybe!

    Mildly adapted from Dr. Sarkar's default version.
    '''

    def __init__(self, unigramFile, bigramFile, countSep='\t', wordSep=' ',
                 alpha=21500, gamma=8.8, kappa=0.6):

        # Define constants
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

        # Compute unigram count
        self.wordCount = {}
        for line in file(unigramFile):
            (word, count) = line.split(countSep)
            utf8key = unicode(word, 'utf-8')
            self.wordCount[utf8key] = [int(count), {}]

        # Get the len of the longest word
        self.maxlen = int(max(map(len, self.wordCount.keys())))
        # Get the number of words in the corpus
        self.N = float(sum([i[0] for i in self.wordCount.itervalues()]))

        # Include beginning of sentence
        self.wordCount["<S>"] = [0, {}]

        # initiate the bigram frequency of frequencies
        # Parse bigram file
        self.bgN = 0
        for line in file(bigramFile):
            (words, count) = line.split(countSep)
            (word1, word2) = words.split(wordSep)
            key1, key2 = unicode(word1, 'utf-8'), unicode(word2, 'utf-8')

            # Special case to find occurrences of <S>
            self.wordCount["<S>"][0] += int(count) if word1 == "<S>" else 0

            # Add bigram info for key2 occurring after key1
            self.wordCount[key1][1][key2] = \
                self.wordCount[key1][1].get(key2, 0) + int(count)
            self.bgN += int(count)

        # self.estimatedProb, self.p0 = self.estimateGoodTuringProbs()

        # update bfFreqofFreq
        self.bgFreqOfFreq = self.setupBGFOF()

        # get Good Turing counts for bigrams
        self.turingCounts = self.setupGoodTuringBigram(self.bgFreqOfFreq)

        # get Katz-Backoff coefficients for all words
        self.allAlpha = defaultdict(lambda: 1)
        for word in self.wordCount.keys():
            self.allAlpha[word] = self.calculateWordAlpha(word)

        return

    def setupBGFOF(self):
        bgFreqOfFreq = defaultdict(int)
        for word1 in self.wordCount.iterkeys():
            for word2 in self.wordCount[word1][1].iterkeys():
                bgnum = self.wordCount[word1][1][word2]
                bgFreqOfFreq[bgnum] += 1

        return bgFreqOfFreq

    def setupGoodTuringBigram(self, bgFreqOfFreq):
        '''
        Setup the interpolation for the good turing bigram counts
        '''

        # grab all FoF info to be used for numpy interpolation
        countsOfCounts = []
        counts = sorted(bgFreqOfFreq.keys())
        for keyCount in counts:
            countsOfCounts.append(bgFreqOfFreq[keyCount])
        counts = np.array(counts)
        countsOfCounts = np.array(countsOfCounts)

        f = interp.interp1d(counts, countsOfCounts)

        # get all the turing counts
        turingCounts = []
        for c in xrange(counts[-1] - 1):
            count = c + 1
            newCount = (count + 1) * f(count + 1) / f(count)
            turingCounts.append(newCount)
        turingCounts.append(counts[-1])

        return turingCounts

    def calculateWordAlpha(self, word):
        '''
        Calculate the Katz-Backoff coefficient for a given word
        '''
        probSum = 0
        wordSuccessors = self.wordCount[word][1]
        wordCount = self.wordCount[word][0]
        for word2 in wordSuccessors.keys():
            probSum += self.getGoodTuringCount(wordSuccessors[word2])
        return 1 - probSum / float(wordCount)

    def getGoodTuringCount(self, bgCount):
        '''
        Return pre-calculated Good Turing count for a given bigram count
        '''
        bgCount = min(len(self.turingCounts), bgCount)
        return self.turingCounts[bgCount - 1]

    def uniProba(self, word, nUnseen, coef=1):
        '''
        Returns the unigram probability for the given word
        '''
        try:
            # return logProb(coef * self.wordCount[word][0]
            #                / (self.N + nUnseen))
            return logProb(coef * self.wordCount[word][0] / self.N)
        except KeyError:
            return logProb(coef * (self.alpha / (self.N + nUnseen))
                           * math.exp(-self.gamma * len(word)))

    def biProba(self, word1, word2, nUnseen):

        '''
        # word2 given word1
        pW1 = self.gtUniProba(word1)

        try:
            #print word2, word1
            #print (self.wordCount[word1][1][word2]/(pW1 * self.bgN)) * theta
            return logProb(self.wordCount[word1][1][word2]/(pW1 * self.bgN))
        except KeyError:
        #print word2, word1
        #print logProb((1-theta)*pW1 * exp(-20 * len(word2)))
            return logProb(pW1 * exp(-self.gamma * len(word2)))
        '''

        # get count of word1, if already zero, use exponential decay
        firstWordCount = 0
        try:
            firstWordCount = self.wordCount[word1][0]
        except KeyError:
            outProb = self.uniProba(word2, nUnseen)
            return outProb

        # get good turing count of bigram
        try:
            bigramInstances = self.wordCount[word1][1][word2]
            turingCount = self.getGoodTuringCount(bigramInstances)
            newProba = logProb(self.kappa * turingCount / firstWordCount)
            return newProba
        except KeyError:
            a = self.allAlpha[word1]
            outProb = self.uniProba(word2, nUnseen, coef=a)
            return outProb

    '''
    def gtUniProba(self, word):
        try:
            return self.estimatedProb[word]
        except KeyError:
            return self.p0 * exp(-self.alpha * len(word))
    '''

    def updateCount(self, word, count):
        '''
        Add [count] instances of [word] to corpus
        '''
        try:
            self.wordCount[word][0] += int(count)
        except:
            self.wordCount[word] = [int(count), {}]

    def updateBigram(self, word1, word2, count):
        '''
        Add [count] instances of bigram to corpus
        '''
        try:
            self.wordCount[word1]
        except:
            self.wordCount[word1] = [int(count), {}]

        try:
            self.wordCount[word1][1][word2] += int(count)
        except:
            self.wordCount[word1][1][word2] = int(count)

    def addSentence(self, sentence):
        '''
        Use a sentence as input to the corpus
        '''
        splitSentence = sentence.split(" ")
        self.updateBigram("<S>", splitSentence[0], 1)

        for ii, word in enumerate(splitSentence[:-1]):
            self.updateCount(word, 1)
            self.updateBigram(word, splitSentence[ii + 1], 1)

        self.updateCount(splitSentence[-1], 1)

        # reupdate important data
        self.bgFreqOfFreq = self.setupBGFOF()
        self.turingCounts = self.setupGoodTuringBigram(self.bgFreqOfFreq)

    '''
    def estimateGoodTuringProbs(self, confidenceLevel=1.96):

        # Compute frequency of frequencies table
        freqOfFreq = self.getFreqOfFreq()

        # Sort counts
        sortedCounts = sorted(freqOfFreq.keys())
        
        # Compute set Z
        Z = self.getZ(sortedCounts, freqOfFreq)
        rs = Z.keys()
        zs = Z.values()

        # Fit a log linear curve to Z
        a, b = self.logLinearRegression(rs, zs)

        rSmoothed = {}
        useY = False
        for r in sortedCounts:
            y = float(r+1) * exp(a*log(r+1) + b) / exp(a*log(r) + b)
            if r+1 not in freqOfFreq:
                useY = True

            if useY == True:
                rSmoothed[r] = y
            else:
                x = (float(r+1) * freqOfFreq[r+1])/ freqOfFreq[r]
                Nr = float(freqOfFreq[r])
                Nr1 = float(freqOfFreq[r+1])
                t = confidenceLevel * sqrt(float(r+1)**2 * (Nr1 / Nr**2) * (1. + (Nr1 / Nr)))
                rSmoothed[r] = x if abs(x - y) > t else y        
                useY = True 

        # Considering just the unigrams
        totalCounts = self.N 
        p0 = freqOfFreq[1] / totalCounts
        gtProbs = {}
        smoothTot = 0.0
        for r, rSmooth in rSmoothed.iteritems():
            smoothTot += freqOfFreq[r ] * rSmooth
        for species in self.wordCount:
            spCount = self.wordCount[species][0]
            gtProbs[species] = (1.0 - p0) * (rSmoothed[spCount] / smoothTot)
        return gtProbs, p0

    def logLinearRegression(self, rs, zs):
        return linalg.lstsq(c_[log(rs), (1,)*len(rs)], log(zs))[0]

    def getZ(self, sortedCounts, freqOfFreq):
        Z = {}
        for jIdx, j in enumerate(sortedCounts):

            # Get previous point i
            i = 0 if jIdx == 0 else sortedCounts[jIdx-1]

            # Get next point k
            k = 2*j - i if jIdx == len(sortedCounts) - 1 else sortedCounts[jIdx+1]

            Z[j] = freqOfFreq[j]/(0.5*(k-i))
        return Z

    def getFreqOfFreq(self):
        """
        This function returns a table containing the
        frequency of frequencies in the corpus
        """
        fof = {}
        counts = [i[0] for i in self.wordCount.itervalues()]
        for c in counts:
            fof[c] = 0
            for val in counts:
                if val == c:
                    fof[c] += 1
        return fof
    '''

    def getNUseen(self, sequence):
        """
        Returns the number of unseen words in a given sentece
        """
        nUnseen = 0
        for endindex in range(len(sequence)):

            lenToCheck = self.maxlen \
                if len(sequence) - self.maxlen > endindex \
                else len(sequence) - endindex - 1

            for i in range(lenToCheck):
                lowInd = endindex + 1
                highInd = endindex + 2 + i
                try:
                    self.wordCount[sequence[lowInd: highInd]]
                except:
                    nUnseen += 1
        return nUnseen


def logProb(prob):
    '''
    temporary workaround to logging zero probabilities
    '''
    if prob > 0:
        return math.log(prob)
    else:
        return -sys.float_info.max


def equal(item1, item2):
    """
    Check if item1 and item2 are equal,
    taking into consideration the log-probability, word and start-position
    """
    return True if item1[0] == item2[0] \
        and item1[1][0] == item2[1][0] \
        and item1[1][1] == item2[1][1] \
        else False


def removeDuplicate(items):
    """
    Remove duplicates in a given list
    """
    newItems = []
    for i1 in items:
        isEqual = [True if equal(i1, i2) else False for i2 in newItems]
        if True not in isEqual:
            newItems.append(i1)
    return newItems


def unfoldEntries(entry):
    """
    Reconstruct, recursively, the best segmentation found
    """
    backPointer = entry[1][2]
    if backPointer != -1:
        return unfoldEntries(backPointer) + [entry[1][0]]
    return [entry[1][0]]


def unigramSegment(sequence, corpus):
    """
    Perform segmentation considering the unigrams probabilities
    """

    # Init Heap
    heap = []
    usequence = unicode(sequence, 'utf-8')
    nUnseen = corpus.getNUseen(usequence)

    # items = [(-corpus.uniProba(usequence[0:i + 1], nUnseen),
    #          (usequence[0:i + 1], 0, -1))
    #          for i in range(corpus.maxlen)]
    items = [(-corpus.uniProba(usequence[0:i + 1], nUnseen),
             (usequence[0:i + 1], 0, -1))
             for i in range(corpus.maxlen)]
    itemsUnique = removeDuplicate(items)
    map(lambda i: heapq.heappush(heap, i), itemsUnique)

    # Iteratively fill in chart
    N = len(usequence)
    chart = N * [()]
    while len(heap) > 0:

        # entry: (log-probability, (word, start_position, back-pointer))
        entry = heapq.heappop(heap)
        endindex = entry[1][1] + len(entry[1][0]) - 1
        preventry = chart[endindex]

        # Check if preventry exists
        if preventry is not () and preventry[0] < entry[0]:
            continue

        # Check previous entry
        chart[endindex] = preventry \
            if preventry is not () and preventry[0] < entry[0]  \
            else entry

        # Add new entries
        newItems = []

        lenToCheck = corpus.maxlen \
            if len(usequence) - corpus.maxlen > endindex \
            else len(usequence) - endindex - 1

        for i in range(lenToCheck):
            low = endindex + 1
            high = endindex + 2 + i
            prevWord = entry[1][0]
            newProb = entry[0] \
                - corpus.uniProba(usequence[low: high], nUnseen)
            newItems.append((newProb,
                            (usequence[low: high], low, entry)))
        newItemsUnique = removeDuplicate(newItems)
        map(lambda i: heapq.heappush(heap, i), newItemsUnique)

    # Print output
    return " ".join(unfoldEntries(chart[-1])), chart[-1][0]

def checkNames(sentence):

    # Frequent names
    names = [unicode("张伟", 'utf-8'),
            unicode("王伟", 'utf-8'),
            unicode("王芳", 'utf-8'),
            unicode("李伟", 'utf-8'),
            unicode("王秀英", 'utf-8'),
            unicode("李秀英", 'utf-8'),
            unicode("李娜", 'utf-8'),
            unicode("张秀英", 'utf-8'),
            unicode("刘伟", 'utf-8'),
            unicode("张敏", 'utf-8'),
            unicode("李静", 'utf-8'),
            unicode("张丽", 'utf-8'),
            unicode("王静", 'utf-8'),
            unicode("王丽", 'utf-8'),
            unicode("李强", 'utf-8'),
            unicode("张静", 'utf-8'),
            unicode("李敏", 'utf-8'),
            unicode("王敏", 'utf-8'),
            unicode("王磊", 'utf-8'),
            unicode("王丽", 'utf-8'),
            unicode("李军", 'utf-8'),
            unicode("刘洋", 'utf-8'),
            unicode("王勇", 'utf-8'),
            unicode("张勇", 'utf-8'),
            unicode("王艳", 'utf-8'),
            unicode("矮丑穷", 'utf-8'),
            unicode("A片", 'utf-8'),
            unicode("阿三", 'utf-8'),
            unicode("棒子", 'utf-8'),
            unicode("版主", 'utf-8'),
            unicode("白富美", 'utf-8'),
            unicode("杯具=悲剧", 'utf-8'),
            unicode("被自杀", 'utf-8'),
            unicode("鄙视", 'utf-8'),
            unicode("标题党", 'utf-8'),
            unicode("变态", 'utf-8'),
            unicode("玻璃", 'utf-8'),
            unicode("菜鸟", 'utf-8'),
            unicode("惨剧", 'utf-8'),
            unicode("李杰", 'utf-8')]

    for name in names:
        pos = sentence.find(name)
        if pos > 0:
            if pos-1 >= 0 and sentence[pos-1] != " ":
                sentence = sentence[:pos] + " " + sentence[pos:]
            if pos+1+ len(name)< len(sentence) and sentence[pos+1+len(name)] != " ":
                sentence = sentence[:pos + 1+ len(name)] + " " + sentence[pos + 1+ len(name):]  

    return sentence

def sanityCheck(sentence):
    '''
    HEURISTICS

    DOES ANYBODY REALLY KNOW HOW TO DO CHINESE REGEX
    '''

    nums = [unicode("３", 'utf-8'),
            unicode("１", 'utf-8'),
            unicode("９", 'utf-8'),
            unicode("８", 'utf-8'),
            unicode("４", 'utf-8'),
            unicode("６", 'utf-8'),
            unicode("５", 'utf-8'),
            unicode("７", 'utf-8'),
            unicode("２", 'utf-8'),
            unicode("０", 'utf-8')]

    num_pres = [unicode("·", 'utf-8'),
                " "]

    midChars = [unicode("：", "utf-8"),
                unicode("·", "utf-8")]

    specialChars = [unicode("日", 'utf-8'),
                    unicode("万", 'utf-8')]

    goodChars = nums + num_pres

    newSentence = ""
    dotColonNum = False
    prevchar = unicode("１", 'utf-8')
    for ii, ch in enumerate(sentence[:-1]):

        # number never preceded by character
        if prevchar not in goodChars and ch in nums:
            newSentence += " "

        # check if we don't already have proper midChar format
        if ch == " " and prevchar in nums and \
           sentence[ii + 1] not in nums:
            dotColonNum = False

        # add spaces after numbers that come after colons and dots
        if dotColonNum is True and ch not in goodChars \
           and ch not in midChars and prevchar in nums:
            newSentence += " "

        # set flag for dots and colons
        if ch in midChars:
            dotColonNum = True
        elif ch not in midChars and ch not in goodChars:
            dotColonNum = False

        # always space between nums and colon
        if prevchar in nums and ch == midChars[0]:
            newSentence += " "

        # never a space between numbers
        if prevchar in nums and ch == " ":
            if sentence[ii + 1] in nums:
                continue
            # no space between numbers and dots
            if sentence[ii + 1] == num_pres[0]:
                continue
            # no space between numbers and special char 0
            if sentence[ii + 1] == specialChars[0]:
                continue
            # no space between numbers and special char 1
            if sentence[ii + 1] == specialChars[1]:
                continue

        # never a space around a dot
        if ch == " " and prevchar == midChars[1]:
            continue
        if ch == " " and sentence[ii + 1] == midChars[1]:
            continue

        # no space between dots and numbers
        if prevchar == num_pres[0] and ch == " ":
            if sentence[ii + 1] in nums:
                continue

        newSentence += ch
        prevchar = ch

    newSentence += sentence[-1]

    return newSentence


if __name__ == "__main__":

    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    if len(sys.argv) > 1:
        alpha = float(opts.alpha)
        gamma = float(opts.gamma)
        beta = float(opts.beta)
        kappa = float(opts.kappa)

        corpus = Corpus(unigramPath, bigramPath, alpha=alpha,
                        gamma=gamma, kappa=kappa)
    else:
        corpus = Corpus(unigramPath, bigramPath)

    # STYLE 1
    # FULL BATCH, STORE SENTENCES, ADAPT ON TOP (100 * BETA)% EXAMPLES

    for ii in range(1):
        all_sentences, all_probs = [], []
        with open(opts.input) as f:
            for line in f:
                outSentence, outProb = unigramSegment(line.strip('\n'), corpus)
                all_sentences.append(outSentence)
                all_probs.append(outProb)

        # find top [100 * beta]% of sentences in terms on probabilities
        all_probs = np.array(all_probs)
        sentenceNum = int(all_probs.size * beta)
        betaIndices = all_probs.argsort()[-sentenceNum:][::-1]

        for bI in betaIndices:
            # corpus.addSentence(sanityCheck(all_sentences[bI]))
            corpus.addSentence(all_sentences[bI])

    # STYLE 3
    # BATCH ADAPTATION, SPLIT INPUT DATA INTO BATCHES

    '''
    batchSize = 5
    all_sentences, all_probs = [], []
    with open(opts.input) as f:
        for i, line in enumerate(f):
            outSentence, _ = unigramSegment(line.strip('\n'), corpus)
            all_sentences.append(outSentence)

            if i % batchSize == 0:
                for s in all_sentences:
                    corpus.addSentence(s)
                all_sentences = []
    '''

    with open(opts.input) as f:
        for line in f:
            outSentence, _ = unigramSegment(line.strip('\n'), corpus)
            outSentence = sanityCheck(outSentence)
            outSentence = checkNames(outSentence)
            print outSentence

    sys.stdout = old
