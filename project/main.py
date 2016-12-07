from decode_FULL import decode
from bleu import bleu, bleu_stats
import numpy as np

numSentences = 100
bleuNum = 10  # number of bleu stats

# open reference file
referenceFile = "./data/all.cn-en.en0"
f = open(referenceFile, "r")

sourceFile1 = "./data/all.cn-en.cn"

allStats = np.empty((numSentences, bleuNum))

for i, nbest in enumerate(decode(sourceFile1, numSentences=numSentences)):

    if i % 50 == 0:
        print "Decoding sentence {} out of {}...".format(i, numSentences)

    ref = f.readline().strip()

    stats = [el for el in bleu_stats(nbest[0], ref)]
    stats = np.array(stats)
    allStats[i, :] = stats

bleuScore = bleu(np.mean(allStats, axis=0))
print "BLEU score: {}".format(bleuScore)
