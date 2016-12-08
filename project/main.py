from decode_FULL import decode
from bleu import bleu, bleu_stats
import numpy as np

numSentences = 5  # number of sentences to decode
bleuNum = 10  # number of bleu stats

# open reference files and specify weights
referenceFiles = ["./data/all.cn-en.en0",
                  "./data/all.cn-en.en1",
                  "./data/all.cn-en.en2",
                  "./data/all.cn-en.en3"]
refs = [open(f, "r") for f in referenceFiles]
refWeights = [0.25, 0.25, 0.25, 0.25]

sourceFile = "./data/all.cn-en.cn"

allStats = [np.empty((numSentences, bleuNum)) for _ in refs]

numIter = 1

# run for a number of iterations to improve BLEU score
for k in range(numIter):

    print
    print "Beginning iteration {}...".format(k)
    print

    # calculate BLEU score across all top translations for all references
    # use current version of feature weights
    for i, nbest in enumerate(decode(sourceFile, numSentences=numSentences)):

        # iterate across references
        for rInd, r in enumerate(refs):

            if i % 50 == 0:
                print "Decoding sentence {} out of {}...".format(i, numSentences)

            ref = r.readline().strip()

            stats = [el for el in bleu_stats(nbest[0], ref)]
            stats = np.array(stats)
            allStats[rInd][i, :] = stats

        # calculate BLEU score
        means = [np.mean(allS, axis=0) for allS in allStats]
        averagedBleu = sum([bleu(m) * w for m, w in zip(means, refWeights)])

        # save sentence and features to nbest file

    # run reranker on nbest file to produce new feature weights

print "BLEU score: {}".format(averagedBleu)
