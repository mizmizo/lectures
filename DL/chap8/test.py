from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle
import numpy as np
import theano
import theano.tensor as T


rng = np.random.RandomState(1234)

def unpickle(file):
    with open(file, 'rb') as f:
        data = cPickle.load(f)
    return data

trn = [unpickle('../cifar_10/data_batch_%d' %i) for i in range(1,6)]
cifar_X_1 = np.concatenate([d['data'] for d in trn]).astype('float32')
cifar_y_1 = np.concatenate([d['labels'] for d in trn]).astype('int32')

tst = unpickle('../cifar_10/test_batch')
cifar_X_2 = tst['data'].astype('float32')
cifar_y_2 = np.array(tst['labels'], dtype='int32')

cifar_X = np.r_[cifar_X_1, cifar_X_2]
cifar_y = np.r_[cifar_y_1, cifar_y_2]

cifar_X = cifar_X / 255.

train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=0.2, random_state=42)

pred_y = homework(train_X, test_X, train_y)

test_y = np.eye(10)[test_y].astype('int32')
print "score: %.3f" % f1_score(np.argmax(test_y, axis=1).astype('int32'), pred_y, average='macro')

