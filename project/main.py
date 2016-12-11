from decode_FULL import decode
from rerank import PRO
from bleu import bleu, bleu_stats, smoothed_bleu
import numpy as np
import models_FULL as models

numSentences = 100  # number of sentences to decode
bleuNum = 10  # number of bleu stats
numIter = 4

# open reference files and specify weights
referenceFiles = ["./data/all.cn-en.en0",
                  "./data/all.cn-en.en1",
                  "./data/all.cn-en.en2",
                  "./data/all.cn-en.en3"]
refs = [open(f, "r") for f in referenceFiles]
scoreWeights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
refWeights = [0.25, 0.25, 0.25, 0.25]

numPhrases = 7
translationModel = "./data/rules_cnt.final.out"
languageModel = "./data/en.gigaword.3g.filtered.dev_test.arpa.gz"
sourceFile = "./data/all.cn-en.cn"
nbestFilePath = "./data/dev.nbest"

allStats = [np.empty((numSentences, bleuNum)) for _ in refs]

# load translation model and language model
tm = models.TM(translationModel, numPhrases)
lm = models.LM(languageModel)

# run for a number of iterations to improve BLEU score
for k in range(numIter):

    print
    print "Beginning iteration {}...".format(k)
    print

    nbestFile = open(nbestFilePath, "w")

    # calculate BLEU score across all top translations for all references
    # use current version of feature weights
    for i, (nbest, feats) in enumerate(decode(sourceFile,
                                              numSentences=numSentences,
                                              rerankWeights=scoreWeights,
                                              lm=lm, tm=tm)):

        if i % 20 == 0:
            print "Decoding {} / {}...".format(i, numSentences)

        # iterate across references
        for rInd, r in enumerate(refs):

            ref = r.readline().strip()

            stats = [el for el in bleu_stats(nbest[0], ref)]
            stats = np.array(stats)
            allStats[rInd][i, :] = stats

        # add sentence and stats to nbest file
        for sentence, sentenceFeats in zip(nbest, feats):

            # grab BLEU score
            bleuScore = bleu([el for el in bleu_stats(sentence, ref)])
            smoothBleu = smoothed_bleu([el for el
                                        in bleu_stats(sentence, ref)])

            nbestFile.write(str(i) + " ||| ")
            nbestFile.write(sentence + " ||| ")
            for feat in sentenceFeats:
                nbestFile.write("{} ".format(feat))
            nbestFile.write(" ||| {}".format(bleuScore))
            nbestFile.write(" ||| {}".format(smoothBleu))
            nbestFile.write("\n")

    # calculate BLEU score
    means = [np.mean(allS, axis=0) for allS in allStats]
    averagedBleu = sum([bleu(m) * w for m, w in zip(means, refWeights)])

    # run reranker on nbest file to produce new feature weights
    theta = PRO(nbestFilePath)
    scoreWeights += theta

    print "BLEU score: {}".format(averagedBleu)
