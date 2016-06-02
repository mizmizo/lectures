from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import theano
import theano.tensor as T

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)

pred_y = homework(train_X, test_X, train_y)

test_y = np.eye(10)[test_y].astype('int32')
print "score: %.3f" % f1_score(np.argmax(test_y, axis=1).astype('int32'), pred_y, average='macro')
