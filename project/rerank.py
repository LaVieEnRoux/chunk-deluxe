#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple, defaultdict
from bleu import bleu_stats, bleu, smoothed_bleu
import random as r
import numpy as np
from scipy.stats import zscore
import subprocess

optparser = optparse.OptionParser()
optparser.add_option("-n", "--train_nbest", dest="train_nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-e", "--entrain", dest="entrain", default=os.path.join("data", "train.en"), help="Train file")
optparser.add_option("-f", "--test_nbest", dest="test_nbest", default=os.path.join("data", "test.nbest"), help="N-best test file")
(opts, _) = optparser.parse_args()


def obtainTranslations(fr_file, phase='train'):
    # From homework 4

    # Call decoder
    if not os.path.isfile('./' + phase +'.translation'):
        out = subprocess.check_output("python ../decoder/decode.py -t ../decoder/data/tm -l ../decoder/data/lm -s 1000 -i " + fr_file, shell=True)
        with open('./' + phase + '.translation', 'w') as f:
            f.writelines(out)
            return out.strip().split('\n')
    else:
        with open('./' + phase + '.translation') as f:
            return f.readlines()

def preprocess():

    # Get estimated translations
    t_sentences_train = obtainTranslations(fr_file= './data/train.fr', phase='train')
    t_sentences_test = obtainTranslations(fr_file= './data/test.fr', phase='test')

    # Preprocess training data
    # Get nbests
    sys.stderr.write("Extracting stats for training ...\n") 
    samples = []

    # Read english sentences
    with open(opts.entrain) as f:
        train_data = f.readlines()

    # Read nbest data
    allSamples = []
    allFeatures = []
    with open(opts.train_nbest) as f:
        allLines = f.readlines()

    for k, line in enumerate(allLines):

        sys.stderr.write(str(int((k+1)/float(len(allLines))*100)) + "%\r")

        # Extract info
        (i, c_sentence, features) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]
        ref_sentence = train_data[int(i)]

        # Compute bleu score
        stats = [el for el in bleu_stats(c_sentence, ref_sentence)]

        # Compute new features
        f1 = len(c_sentence.strip().split(' '))
        features.append(f1)

        # Decode feature
        t_stats = [el for el in bleu_stats(c_sentence, t_sentences_train[int(i)])]
        features.append(smoothed_bleu(t_stats))

        allFeatures.append(features)
        allSamples.append([str(i), c_sentence, str(bleu(stats)), str(smoothed_bleu(stats))])

    # Normalize and include higher order terms
    normFeatures = np.array(allFeatures)
    normFeatures = zscore(np.array(allFeatures), axis=0)
    normFeatures = np.concatenate([normFeatures, normFeatures**2, normFeatures**3], axis=1)

    with open("mod_train.nbest", 'w') as f:
        for k, (i, c_sentence, str_bleu, str_smoothed_bleu) in enumerate(allSamples):
            str_features = " ".join([str(ft) for ft in normFeatures[k]])
            f.write(" ||| ".join([i, c_sentence, str_features, str_bleu, str_smoothed_bleu]) + '\n')

    # Preprocess test data
    # Get nbests
    sys.stderr.write("Extracting stats for test ...\n") 
  
    # Read nbest data
    with open(opts.test_nbest) as f:
        allLines = f.readlines()

    # Preprocess test data
    allSamples = []
    allFeatures = []
    for k, line in enumerate(allLines):

        # Keep track
        sys.stderr.write(str(int((k+1)/float(len(allLines))*100)) + "%\r")

        # Get info
        (i, c_sentence, features) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]

        # Create new features
        f1 = len(c_sentence.strip().split(' '))
        features.append(f1)

        # Decode feature
        t_stats = [el for el in bleu_stats(c_sentence, t_sentences_test[int(i)])]
        features.append(smoothed_bleu(t_stats))

        # Accumulate
        allFeatures.append(features)
        allSamples.append([str(i), c_sentence])

    # Normalize and include higher order terms
    normFeatures = np.array(allFeatures)
    normFeatures = zscore(np.array(allFeatures), axis=0)
    normFeatures = np.concatenate([normFeatures, normFeatures**2, normFeatures**3], axis=1)
    # WE CANNOT USE TEST.NBEST ??????!!!!
    with open("mod_test.nbest", 'w') as f:
        for k, (i, c_sentence) in enumerate(allSamples):

            str_features = " ".join([str(ft) for ft in normFeatures[k]])
            f.write(" ||| ".join([i, c_sentence, str_features]) + '\n')
    return

def splitting():
    return

def PRO():

    # Params
    epochs = 150
    tau = 5000
    alpha = 0.1
    xi = 500
    eta = 0.001
    batch_size = 50

    # Check if dataset is preprocessed 
    if not os.path.isfile('./mod_test.nbest') or not os.path.isfile('./mod_train.nbest'):
        preprocess()
        
    # Get nbests
    sys.stderr.write("Extracting info ...\n") 
    nbests = defaultdict(list)

    # Read nbest data
    with open('./mod_train.nbest') as f:
        allLines = f.readlines()

    for k, line in enumerate(allLines):

        sys.stderr.write(str(int((k+1)/float(len(allLines))*100)) + "%\r")

        # Extract info
        (i, c_sentence, features, bleu, smoothed_bleu) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]

        # Compute bleu score
        nbests[int(i)] += [(c_sentence, np.array(features), float(bleu), float(smoothed_bleu))]

    # Init theta
    features_num = nbests[0][0][1].shape[0] 
    theta = np.array([.0 for _ in xrange(features_num)])
    sys.stderr.write("\nTuning parameters ...\n")
    for k in xrange(epochs):
        mistakes = 0
    
        for l, nbest in enumerate(nbests.itervalues()):

            # Get sample
            sample = []
            score = []

            # Must have at least two elements for sampling
            for j in xrange(tau):

                if len(nbest) >= 2:

                    idxs = [r.choice(xrange(len(nbest))) for _ in range(2)]
                    s = [nbest[i] for i in idxs]

                    ## MAYBE WE DONT NEED THIS
                    #nbests[l] = [nbest[i] for i in range(len(nbest)) if i not in idxs]

                    fabs_diff = abs(s[0][3] - s[1][3])
                    if fabs_diff > alpha:

                        score.append(fabs_diff)
                        if s[0][3] > s[1][3]:
                            sample += [(s[0], s[1])]
                        else:
                            sample += [(s[1], s[0])]
            
            if len(sample) > 0:
                sorted_sample = [el[0] for el in sorted(zip(sample, score), key=lambda x:x[1])]

                # Use the top xi samples
                update = []
                for count, (s1, s2) in enumerate(sorted_sample[:xi]):
                    if np.dot(theta, s1[1]) <= np.dot(theta, s2[1]):
                        mistakes += 1
                        if (count+1)%batch_size == 0:
                            # Update
                            if len(update) > 0:
                                theta += eta * np.mean(np.array(update), axis=0)
                                update = []
                        else:
                            # Accumulate
                            update.append(s1[1] - s2[1])

        sys.stderr.write("Iter: %d Mistakes: %d\r" % (k+1, mistakes) )

    sys.stderr.write("\nWeights:\n")
    sys.stderr.write("\n".join([str(t) for t in theta]))
    sys.stderr.write("\n")

    print "\n".join([str(t) for t in theta])
    return

if __name__ == '__main__':
    PRO()
