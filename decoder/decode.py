#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple, Counter
import numpy as np
from itertools import combinations

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=13, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int",
                     help="Maximum stack size (default=8)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-d", "--distanceWeight", dest="distanceWeight",
                     default=0.92, type="float", help="Weight for distance decoding")
optparser.add_option("-x", "--languageModelWeight", dest="lmw", default=0.91,
                     type="float")
optparser.add_option("-y", "--translationModelWeight", dest="tmw", default=1.0,
                     type="float")
optparser.add_option("-z", "--distanceModelWeight", dest="dmw", default=1.0,
                     type="float")

opts = optparser.parse_args()[0]

def estimate_cost(logprob, lm, phrase):

    lm_state = (phrase[0], )
    _, logprob = lm.score(lm_state, ())

    if len(phrase) >= 2:
        _, word_logprob = lm.score(lm_state, (phrase[1], ))
        logprob += word_logprob
        lm_state += (phrase[1], )

        for word in phrase[2:]:
            (lm_state, word_logprob) = lm.score(lm_state, (word, ))
            logprob += word_logprob
    return logprob

def estimate_cost_table(tm, lm, f):
    future_cost = {}
    for i, f_w_1 in enumerate(f):
        acc = ()
        for f_w_2 in f[i:]:
            acc += (f_w_2, )
            try:
                future_cost[acc] = np.max([estimate_cost(phrase.logprob, lm, phrase.english.split()) for phrase in tm[acc]])
            except:
                future_cost[acc] = -99999
    return future_cost

def get_fc(future_cost, locs, f):

    cost = 0.0
    subPhrase = ()
    for k, l in enumerate(locs):
        if l == 0:
            subPhrase += (f[k], )
        else:
            if len(subPhrase) > 0:
                cost += future_cost[subPhrase]
            subPhrase = ()
    
    # In case locs finish with zeros
    if len(subPhrase) > 0:
        cost += future_cost[subPhrase]
    return cost

def all_comb(el):
    # Generate all combinations up to k
    all_comb = []
    for i in range(len(el)):
        all_comb += combinations(el, i+1)
    return all_comb

def getPhraseListV1(f, locs, MAXLEN=10):
    '''
    This code is for generating a set of proposals for possible
    consecutive sentences to translate
    '''

    # locs is a vector like (0, 1, 1, ....)
    # it defines which phrases we can afford to add
    phraseList = []
    locList = []

    # # iterate through locs and find all subsets of f that work
    p_locs = np.where(np.array(locs)==0)[0]

    for i in range(len(locs)):

        # For each zero
        if locs[i] == 0:
            sup = min(len(locs), i+MAXLEN)
            combs = []; acc = []

            # Look ahead until sup or already decoded word
            for j in range(i, sup):
                if locs[j] == 0:
                    acc.append(j)
                    combs.append(list(acc))
                else:
                    break

            # Create new phrases
            for comb in combs:
                newLocs = [locs[k] if k not in comb else 1 for k in range(len(locs))]
                newPhrase = tuple([f[k] for k in comb])

                if newPhrase not in phraseList:
                    phraseList.append(newPhrase)
                    locList.append(newLocs)

    return phraseList, locList

def getPhraseListV2(f, locs, MAXLEN=7, MAXLOOKAHEAD=7):
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

    future_cost_table = estimate_cost_table(tm, lm, f)
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, "
                            + "phrase, decodeLocs, prevPhraseEnd, future_cost") 
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, [0] * len(f),
                                    0, -99999)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    winner = []

    for i, stack in enumerate(stacks[:-1]):
        for pos, h in enumerate(sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]): # prune\
            #print [len(s.keys()) for s in stacks]
            #print h.logprob, h.future_cost
            # generate list of possible words for new hypotheses
            if winner != [] and h.logprob < winner.logprob: # prune
                #print "%d out of %d at %d" % (pos, len(stack.keys()), i) 
                break

            newPhrases, newLocs = getPhraseListV2(f, h.decodeLocs, MAXLEN=int(len(f)), MAXLOOKAHEAD=11)
            for newPhrase, newLoc in zip(newPhrases, newLocs):

                # we don't need an entire damn indentation level
                # just for this conditional
                if newPhrase not in tm:
                    continue
                
                for phrase in tm[newPhrase]:

                    # calculate and add translation model score
                    logprob = h.logprob + phrase.logprob * opts.tmw
                    lm_state = h.lm_state

                    # calculate and add language model score
                    lm_logprob = 0.0
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        lm_logprob += word_logprob 
                    logprob +=  lm_logprob * opts.lmw

                    # calculate and add distance score
                    newPhraseLocs = [a - b for a, b in zip(newLoc, h.decodeLocs)]
                    decodedInd = [i for i, v in enumerate(newPhraseLocs) if v == 1]
                    phraseStart = decodedInd[0]
                    distance = abs(h.prevPhraseEnd + 1 - phraseStart)
                    # distance += abs(len(phrase.english.split()) - len(newPhrase))
                    logprob += np.log(opts.distanceWeight) * distance * opts.dmw

                    # j is the number of matched words
                    j = newLoc.count(1)

                    # add to hypothesis stack
                    logprob += lm.end(lm_state) if j == len(f) else 0.0
                    cost = get_fc(future_cost_table, newLoc, f)
                    new_hypothesis = hypothesis(logprob, lm_state, h, phrase, newLoc, decodedInd[-1], cost)

                    # Check winner
                    if j == len(f):
                        if winner == []:
                            winner = new_hypothesis
                        elif winner.logprob < new_hypothesis.logprob:
                            winner = new_hypothesis
                    else:        
                        # Add to stack
                        if lm_state not in stacks[j] \
                        or stacks[j][lm_state].logprob < logprob: # second case is recombination
                            stacks[j][lm_state] = new_hypothesis

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)

if opts.verbose:
    def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
        (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
