#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple, Counter
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=11, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=4000, type="int",
                     help="Maximum stack size (default=8)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-d", "--distanceWeight", dest="distanceWeight",
                     default=0.92, type="float", help="Weight for distance decoding")
optparser.add_option("-lm", "--languageModelWeight", dest="lmw", default=0.33,
                     type="float")
optparser.add_option("-tm", "--translationModelWeight", dest="tmw", default=0.33, 
                     type="float")
optparser.add_option("-dm", "--distanceModelWeight", dest="dmw", default=0.33,
                     type="float")

opts = optparser.parse_args()[0]

def getPhraseList(f, locs, MAXLEN=7, MAXLOOKAHEAD=7):
    '''
    This code is for generating a set of proposals for possible
    consecutive sentences to translate
    '''

    # locs is a vector like (0, 1, 1, ....)
    # it defines which phrases we can afford to add
    phraseList = []
    locList = []

    # iterate through locs and find all subsets of f that work
    numFree = 0
    lookahead = 0
    for i, v in enumerate(locs):
        if v == 0:

            if lookahead == 0:
                lookahead += 1

            # prevents us from looking back too far
            if numFree < MAXLEN:
                numFree += 1

            if numFree <= MAXLEN and numFree > 0:

                # step backwards and add phrases
                for j in xrange(1, numFree + 1):
                    phraseList.append(f[i - j + 1: i + 1])
                    newLoc = [0] * len(f)
                    newLoc[i - j + 1: i + 1] = [1] * j
                    loc = [x + w for x, w in zip(locs, newLoc)]
                    locList.append(loc)

        else:
            # reset
            numFree = 0

        # don't look too far ahead!
        if lookahead > MAXLOOKAHEAD:
            break
        if lookahead > 0:
            lookahead += 1

    return phraseList, locList


tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, "
                            + "phrase, decodeLocs, prevPhraseEnd") 
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, [0] * len(f),
                                    0)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis

    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]: # prune

            # sys.stderr.write("Hypothesis: {}\n".format(h))

            # generate list of possible words for new hypotheses
            newPhrases, newLocs = getPhraseList(f, h.decodeLocs)

            '''
            sys.stderr.write("\nCurrent locations: {}\n".format(h.decodeLocs))
            sys.stderr.write("French phrase: {}\n".format(f))
            sys.stderr.write("New phrases: {}\n".format(newPhrases))
            sys.stderr.write("Corresponding locations: {}\n".format(newLocs))
            '''

            for newPhrase, newLoc in zip(newPhrases, newLocs):

                # we don't need an entire damn indentation level
                # just for this conditional
                if newPhrase not in tm:
                    continue

                # sys.stderr.write("Current french candidate: " +
                #                  str(fr) + "\n")
                for phrase in tm[newPhrase]:

                    # calculate and add translation model score
                    logprob = h.logprob + phrase.logprob * opts.tmw
                    lm_state = h.lm_state

                    # calculate and add language model score
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob * opts.lmw

                    # calculate and add distance score
                    newPhraseLocs = [a - b for a, b in zip(newLoc,
                                                           h.decodeLocs)]
                    decodedInd = [i for i, v in enumerate(newPhraseLocs)
                                  if v == 1]
                    phraseStart = decodedInd[0]
                    distance = abs(phraseStart - h.prevPhraseEnd - 1)
                    logprob += np.log(opts.distanceWeight) * distance * opts.dmw

                    # j is the number of matched words
                    j = newLoc.count(1)

                    # add to hypothesis stack
                    logprob += lm.end(lm_state) if j == len(f) else 0.0
                    new_hypothesis = hypothesis(logprob, lm_state, h, phrase,
                                                newLoc, decodedInd[-1])

                    if lm_state not in stacks[j] \
                    or stacks[j][lm_state].logprob < logprob: # second case is recombination
                        stacks[j][lm_state] = new_hypothesis

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)


if opts.verbose:
    def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
        (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
