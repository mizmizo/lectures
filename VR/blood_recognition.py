
# coding: utf-8

# # SCENE RECOGNITION

# In[ ]:

def do_work(train_X, test_X, train_y):
    train_y = np.eye(10)[train_y].astype('int32')
    train_X = train_X.reshape((train_X.shape[0], 3, 36, 64))
    test_X = test_X.reshape((test_X.shape[0], 3, 36, 64))

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1)
    
    def gcn(x):
        mean = np.mean(x, axis=(1,2,3), keepdims=True)
        std = np.std(x, axis=(1,2,3), keepdims=True)
        return (x - mean)/std
    
    class ZCAWhitening:
        def __init__(self, epsilon=1e-2):
            self.epsilon = epsilon
            self.mean = None
            self.ZCA_matrix = None
    
        def fit(self, x):
            x = x.reshape(x.shape[0],-1)
            self.mean = np.mean(x,axis=0)
            x -= self.mean
            print x.shape
            cov_matrix = np.dot(x.T, x)/x.shape[0]
            print cov_matrix.shape
            A, d, _ = np.linalg.svd(cov_matrix)
            self.ZCA_matrix = np.dot(np.dot(A,np.diag(1.0 / np.sqrt(d + self.epsilon))),A.T)
            
        def transform(self, x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1)
            x -= self.mean
            x = np.dot(x,self.ZCA_matrix.T)
            return x.reshape(shape)
    
    class BatchNorm:
        #- Constructor
        def __init__(self, shape, epsilon=np.float32(1e-3)):
            self.shape = shape
            self.epsilon = epsilon
        
            self.gamma = theano.shared(np.ones(self.shape, dtype="float32"), name="gamma")
            self.beta = theano.shared(np.zeros(self.shape, dtype="float32"), name="beta")
            self.params = [self.gamma, self.beta]
        
        #- Forward Propagation
        def f_prop(self, x):
            if x.ndim == 2:
                mean = T.mean(x, axis=0, keepdims=True)# WRITE ME
                std = T.sqrt(T.var(x, axis=0, keepdims=True) + self.epsilon)# WRITE ME
            elif x.ndim == 4:
                mean = T.mean(x, axis=(0,2,3), keepdims=True)# WRITE ME (HINT : ndim=4のときはフィルタの次元でも平均をとる)
                std = T.sqrt(T.var(x, axis=(0,2,3), keepdims=True) + self.epsilon)# WRITE ME
        
            normalized_x = (x - mean) / std# WRITE ME
            self.z = self.gamma * normalized_x + self.beta# WRITE ME
            return self.z
    
    class Conv:
        #- Constructor
        def __init__(self, filter_shape, function=lambda x: x, border_mode="valid", subsample=(1, 1)):
            self.function = function
            self.border_mode = border_mode
            self.subsample = subsample
        
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        
            # Xavier
            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (fan_in + fan_out)),
                        high=np.sqrt(6. / (fan_in + fan_out)),
                        size=filter_shape
                    ).astype("float32"), name="W")
            self.b = theano.shared(np.zeros((filter_shape[0],), dtype="float32"), name="b")
            self.params = [self.W,self.b]
        
        #- Forward Propagation
        def f_prop(self, x):
            conv_out = conv2d(x, self.W, border_mode=self.border_mode, subsample=self.subsample)
            self.z = self.function(conv_out + self.b[np.newaxis, :, np.newaxis, np.newaxis])
            return self.z
    
    class Pooling:
        #- Constructor
        def __init__(self, pool_size=(2,2), padding=(0,0), mode='max'):
            self.pool_size = pool_size
            self.mode = mode
            self.padding = padding
            self.params = []
        
        #- Forward Propagation
        def f_prop(self, x):
            return pool.pool_2d(input=x, ds=self.pool_size, padding=self.padding, mode=self.mode, ignore_border=True)
        
    class Flatten:
        #- Constructor
        def __init__(self, outdim=2):
            self.outdim = outdim
            self.params = []

        #- Forward Propagation
        def f_prop(self,x):
            return T.flatten(x, self.outdim)   
    
    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function

            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (in_dim + out_dim)),
                        high=np.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim,out_dim)
                    ).astype("float32"), name="W")

            self.b =  theano.shared(np.zeros(out_dim).astype("float32"), name="b")
            self.params = [ self.W, self.b ]
        
        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z
        
    class Activation:
        #- Constructor
        def __init__(self, function):
            self.function = function
            self.params = []
    
        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z
        
    def sgd(params, g_params, eps=np.float32(0.3)):
        updates = OrderedDict()
        for param, g_param in zip(params, g_params):
            updates[param] = param - eps*g_param
        return updates
    
    activation = T.nnet.relu

    layers = [                               # (チャネル数)x(縦の次元数)x(横の次元数)
        Conv((32, 3, 3, 3)),                 #   3x90x160 ->  32x34x62
        BatchNorm((32, 34, 62)),
        Activation(activation),
        Pooling((2, 2)),                     #  32x34x62 ->  32x17x31
        Conv((64, 32, 3, 3)),                #  32x17x31 ->  64x15x29
        BatchNorm((64, 15, 29)),
        Pooling((2, 2)),                     #  64x15x29 ->  64x 7x 14
        Conv((128, 64, 3, 3)),               #  64x 7x 14 -> 128x 5x 12
        BatchNorm((128, 5, 12)),
        Activation(activation),
        Pooling((2, 2)),                     # 128x 5x 12 -> 128x 2x 6
        Flatten(2),
        Layer(128*2*6, 256, activation),
        Layer(256, 10, T.nnet.softmax)
    ]
    
    x = T.ftensor4('x')
    t = T.imatrix('t')

    params = []
    layer_out = x
    for layer in layers:
        params += layer.params
        layer_out = layer.f_prop(layer_out)

    y = layers[-1].z

    cost = T.mean(T.nnet.categorical_crossentropy(y, t))

    g_params = T.grad(cost, params)
    updates = sgd(params, g_params)

    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')
    
    #zca = ZCAWhitening()
    #zca.fit(gcn(train_X))
    #zca_train_X = zca.transform(gcn(train_X))
    #zca_train_y = train_y[:]
    #zca_valid_X = zca.transform(gcn(valid_X))
    #zca_valid_y = valid_y[:]
    #zca_test_X  = zca.transform(gcn(test_X))
    
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    for epoch in xrange(100):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            cost = train(train_X[start:end], train_y[start:end])
        print 'Training cost: %.3f' % cost
        valid_cost, valid_pred_y = valid(valid_X, valid_y)
        print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), valid_pred_y, average='binary'))
    
    pred_y = test(test_X)
    
    return pred_y


# In[ ]:

from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(1234)

def loading(file):
    data = np.load(file)
    return data

data_list = [[loading('data/hellsing_01_%d_%d.npy' %(j, i)) for i in range(0,8)] for j in range(0,13)]
data_X = np.concatenate([d for d in data_list], axis = 1).astype('float32')
data_X = data_X.reshape(1, data_X.shape[0] * data_X.shape[1], data_X.shape[2])[0]
label_list = [loading('label/hellsing_01_%d.npy' %i) for i in range(0,13)]
data_y = np.concatenate([d for d in label_list]).astype('int32')

data_X = data_X / 255.

train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=52)

pred_y = do_work(train_X, test_X, train_y)

test_y = np.eye(10)[test_y].astype('int32')
print "score: %.3f" % f1_score(np.argmax(test_y, axis=1).astype('int32'), pred_y, average='binary')


# In[ ]:

from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
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


def load_cifar():
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

    return (train_X, test_X, train_y, test_y)

def check_homework():
    train_X, test_X, train_y, test_y = load_cifar()
    pred_y = homework(train_X, test_X, train_y)
    return f1_score(test_y, pred_y, average='macro')

if 'homework' in globals():
    result = check_homework()
    
    print "No Error Occured!"

