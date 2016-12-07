from decode_FULL import decode
from bleu import bleu, bleu_stats

# open reference file
referenceFile = "./data/all.cn-en.en0"
f = open(referenceFile, "r")

sourceFile1 = "./data/all.cn-en.cn"

for nbest in decode(sourceFile1, numSentences=2):

    ref = f.readline().strip()

    print "Reference:"
    print ref

    print
    print "Translations:"
    for trns in nbest:
        print trns
    print
