#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import numpy as np
import itertools


def parse_params():
  optparser = optparse.OptionParser()
  optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
  optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
  optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
  optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
  optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
  optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
  (opts, _) = optparser.parse_args()
  f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
  e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

  if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)
  return opts, f_data, e_data

def get_bitext(f_data, e_data, opts):
  return [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


def myLog(x):
  x = np.clip(x, 10E-12, 10E12)
  return np.log(x)


def EM(bitext):
  
  # French vocabulary size
  V_f = []
  [V_f.extend(f) for (f, e) in bitext]
  Vf_size = len(set(V_f))
  Vf_total = len(V_f)

  # English vocab
  V_e = []
  [V_e.extend(e) for (f, e) in bitext]
  Ve_size = len(set(V_e))
  Ve_total = len(V_e)

  sentenceNum = len(bitext)

  nullWeight = 1. / Vf_size + 0.15
  # nullWeight = 0

  # Init t_k and initialize variables for better initialization
  t_k = defaultdict(lambda:1.0/Vf_size)
  fe_num = defaultdict(int)
  f_num = defaultdict(int)
  e_num = defaultdict(int)
  total_f = 0
  total_fe = 0
  iters = 5

  
  # set up LLR
  for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
      f_num[f_i] += 1

      for e_j in set(e):
          if (f_i, e_j) not in fe_num:
            total_fe += 1
          fe_num[(f_i, e_j)] += 1
          e_num[e_j] += 1

  # calculate LLR normalization term
  # AND use LLR to initialize the t_k
  sys.stderr.write("Renormalizing translation, probabilities...\n")
  largest = 0
  ii = 0
  for e_j in e_num.iterkeys():

    if ii % 200 == 0:
      sys.stderr.write("Word: {} / {}\n".format(ii, Ve_size))
    ii += 1

    e_prob = e_num[e_j] / float(sentenceNum)
    llr_sum = 0

    for f_i in f_num.iterkeys():
      if (f_i, e_j) in fe_num:
        f_prob = f_num[f_i] / float(sentenceNum)
        fe_prob = fe_num[(f_i, e_j)] / float(sentenceNum)

        # sum across t/s variables as in the paper
        # sys.stderr.write("Probs: {}, {}, {}".format(f_prob, e_prob, fe_prob))
        llr_f_e = fe_num[(f_i, e_j)] * myLog(fe_prob / (f_prob * e_prob))
        llr_nf_e = (e_num[e_j] - fe_num[(f_i, e_j)]) \
          * myLog((e_prob - fe_prob) / ((1 - f_prob) * e_prob))
        llr_f_ne = (f_num[f_i] - fe_num[(f_i, e_j)]) \
          * myLog((f_prob - fe_prob) / ((1 - e_prob) * f_prob))
        llr_nf_ne = (sentenceNum - fe_num[(f_i, e_j)]) \
          * myLog((1 - fe_prob) / ((1 - e_prob) * (1 - f_prob)))

        llr = llr_f_e + llr_nf_e + llr_f_ne + llr_nf_ne

        t_k[(f_i, e_j)] = llr

        llr_sum += llr

    largest = max(llr_sum, largest)

  # set llr and normalize
  for (f_i, e_j) in fe_num.iterkeys():
    t_j[(f_i, e_j)] /= float(largest)

  '''
  for (f_i, e_j) in fe_num.iterkeys():
    f_prob = f_num[f_i] / float(Vf_total)
    e_prob = e_num[e_j] / float(Ve_total)
    fe_prob = fe_num[(f_i, e_j)]

    if fe_prob > (f_prob * e_prob):
      llr = fe_num[(f_i, e_j)] * fe_prob / (f_prob * e_prob)
      t_k[(f_i, e_j)] = llr / float(largest)
    else:
      # t_k[(f_i, e_j)] /= float(largest)
      t_k[(f_i, e_j)] = 0
  '''

  sys.stderr.write("Done renormalizing\n")
 
  # clear memory
  f_num.clear()
  fe_num.clear()
  e_num.clear()

  # Repeat until convergence
  for k in range(1, iters+1):
    
    # Init all counts to zero
    e_count = defaultdict(int)
    fe_count = defaultdict(int)
    sys.stderr.write("\nEpoch %i \n" % (k))

    # For each (f,e) in D
    for (n, (f, e)) in enumerate(bitext):

      # For each f_i in f
      for f_i in set(f):
        
        # Normalization term
        Z = np.sum(t_k[(f_i, e_j)] for e_j in set(e))

        # For each e_j in e
        for e_j in set(e):
          c = t_k[(f_i, e_j)]/Z
          fe_count[(f_i, e_j)] += c
          e_count[e_j] += c

      if n % 1000 == 0:
        sys.stderr.write(".")

    # Set new parameters t_k
    for (f, e) in fe_count.keys():
      t_k[(f, e)] = (fe_count[(f, e)]) / (e_count[e])

  # Decoding the best alignment
  sys.stderr.write("\nDecoding ... \n")
  for (f, e) in bitext:
    for (i, f_i) in enumerate(f):

      bestp = 0
      bestj = 0
      beste = None
      for (j, e_j) in enumerate(e):
        if t_k[(f_i, e_j)] > bestp:
          bestp = t_k[(f_i, e_j)]
          bestj = j
          beste = e_j
      if nullWeight > bestp:
        # align with null
        continue
      else:
        # Print alignment
        sys.stdout.write("%i-%i " % (i, bestj))
    sys.stdout.write("\n")
  return 

def print_output(bitext, dice):
  for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
      for (j, e_j) in enumerate(e):
        if dice[(f_i,e_j)] >= opts.threshold:
          sys.stdout.write("%i-%i " % (i,j))
    sys.stdout.write("\n")
  return

def align(bitext):

  # Start training
  sys.stderr.write("Training with EM ... \n")
  EM(bitext)

  return

if __name__ == '__main__':

  # Parse parameters
  opts, f_data, e_data = parse_params()
  bitext = get_bitext(f_data, e_data, opts)

  # Align french and english sentences
  align(bitext)
