"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the
      feat_vec dictionary which can happen if
      \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag,
      there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
from collections import defaultdict
import numpy as np
import subprocess

def perc_train(train_data, tagset, numepochs, model_file, batch_size=10):

    feat_vec = defaultdict(int)
    # please limit the number of iterations of training to n iterations

    # set default tag to "NP"
    default_tag = "B-NP"

    initialDelta = 1
    decayRate = 0.8
    prev_acc = 0.0
    iter_counter = 0

    # iterate through epochs
    for epoch in range(numepochs):
        
        delta = initialDelta * (decayRate ** epoch)

        # iterate through minibatches
        for start_pos in range(0, len(train_data), batch_size):

            # define minibatch indices
            end_pos = start_pos+batch_size
            batch_ids = range(start_pos, end_pos) if end_pos < len(train_data) else range(start_pos, len(train_data)) + range(end_pos-len(train_data))
            np.random.shuffle(batch_ids)
            batch_data = [train_data[i] for i in batch_ids]

            # define weights for minibatch
            b_w = defaultdict(int)

            # reset error counter
            errorNum = 0

	        # iterate through sentences in minibatch
            for sentence_data in batch_data:

                labeled_list = sentence_data[0]
                feat_list = sentence_data[1]

                # grab predicted tag based on current feat_vec
                pred = perc.perc_test(feat_vec, labeled_list, feat_list, tagset,
	                                  default_tag)

	            # check if tag is correct
                true = [s.split()[-1] for s in labeled_list]
                comparisons = [t == p for t, p in zip(true, pred)]
                feat_index = 0

                # if tag is not correct, update weights
                for i, word in enumerate(labeled_list):

                    (feat_index, wordFeats) = \
                        perc.feats_for_word(feat_index, feat_list)

                    if comparisons[i] is False:

                        errorNum += 1

                        # shift weights for correct/incorrect tags
                        for f in wordFeats[:-1]:
                            b_w[f, true[i]] += delta
                            b_w[f, pred[i]] -= delta

                        # update bigram feature weight too
                        if i == 0:
                            b_w["B:B_-1", true[i]] += delta
                            b_w["B:B_-1", pred[i]] -= delta
                        else:
                            b_w["B:" + true[i - 1], true[i]] += delta
                            b_w["B:" + pred[i - 1], pred[i]] -= delta

            # update all weights
            for k in b_w.iterkeys():
                feat_vec[k] += float(b_w[k])/float(batch_size)

            # Perform Validation
            if iter_counter % 50 == 0:
                print >>sys.stderr, "\nIter: " + str(iter_counter) + " Epoch: " + str(epoch)
                prev_acc = validate_model(feat_vec, model_file, prev_acc)
            iter_counter+=1

            #print >>sys.stderr, "Errors at epoch {}: {}".format(epoch, errorNum)

    return feat_vec

def validate_model(feat_vec, model_file, prev_acc):

    print >>sys.stderr, "********* Validating Model **********"
    perc.perc_write_to_file(feat_vec, "data/aux.model")
    subprocess.check_output("python perc.py -m data/aux.model > output", shell=True)
    out = subprocess.check_output("python score-chunks.py -t output | grep -w 'Score'", shell=True)
    acc = float(out.split(" ")[1])

    # Save best model
    if acc > prev_acc:
        print >>sys.stderr, "New Model Found\nValidation Score: " + str(acc)
        subprocess.check_output("cp data/aux.model "+ model_file.split(".")[0] +"_best.model", shell=True)
        return acc
    return prev_acc


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile",
                         dest="tagsetfile",
                         default=os.path.join("data", "tagset.txt"),
                         help="tagset that contains all the labels "
                         + "produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", 
                         dest="trainfile",
                         default=os.path.join("data", "train.txt.gz"),
                         help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile",
                         dest="featfile",
                         default=os.path.join("data", "train.feats.gz"),
                         help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs",
                         dest="numepochs", default=int(11),
                         help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile",
                         dest="modelfile",
                         default=os.path.join("data", "default.model"),
                         help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."

    # Start Training
    perc_train(train_data, tagset, 20, opts.modelfile)#int(opts.numepochs))

