#--- Linear Regression
#- Step1. Define Symbolic / Shared Variables
x, t = T.fvector("x"), T.fvector("t") #input

W = theano.shared(rng.uniform(low=-0.08, high=0.08, size=(5, 3)).astype('float32'), name="W") #variables that are shared over iteration: weight, bias
b = theano.shared(np.zeros(3), name="b")

print type(W)
print W.get_value()
print 
print type(b)
print b.get_value()
#- Step2. Define graph
y = T.dot(x, W) + b

cost = T.sum((y - t)**2) #Cost function

gW, gb = T.grad(cost=cost, wrt=[W, b]) # Take gradient

updates = OrderedDict({W: W-0.01*gW, b: b-0.01*gb}) # Set update expression in OrderedDict


#- Step3. Compile graph
f = theano.function(inputs=[x, t], outputs=[cost, gW, gb], 
                    updates=updates, allow_input_downcast=True)

#- Step4. Run!!
for epoch in xrange(5):
    cost, gW, gb = f([-2., -1., 1., 2., 3.], [.4, .3, .5])
    print "epoch:: %d, cost:: %.3f"%(epoch, cost)

#=========

def identity(a):
    return a

a = T.iscalar('a')
f = theano.function(inputs=[a], outputs=a)

print identity(1)
print f(1)def identity(a):
    return a

a = T.iscalar('a')
f = theano.function(inputs=[a], outputs=a)

print identity(1)

#=========

W = theano.shared(np.array([0., 1., 2., 3., 4.]).astype("float32"), name="W")

print W
print W.get_value()

W.set_value(np.array([0., 2., 2., 3., 4.]).astype("float32"))
print W.get_value()

#=========

## Symbolic Variables 
a = T.iscalar("a")  # integer
b = T.fscalar("b")  # float scalar

x = T.fvector("x")  # float vector
X = T.fmatrix("X")  # float matrix
#Y = T.tensor3("Y")

## Shared Variable, store variables on cpu/gpu memory
W = theano.shared(np.array([0., 1., 2., 3., 4.]).astype("float32"), name="W")
bias = theano.shared(np.float32(5), name="bias")

# Get Value from shared variable
print W.get_value() 

## Define symbolic graph
c = a + b
y = T.dot(x, W) + bias

## Print symbolic graph
print theano.pp(y)

##  Advanced:: You can replace some parts of computation graph with different variable
d = theano.clone(output=c, replace={b: y}) #replace "b" with "y"
print theano.pp(d)

#==============

#- Compile symbplic graph into callable functions
add = theano.function(inputs=[a, b], outputs=c)
linear = theano.function(inputs=[x], outputs=y)

#- Call Functions
print add(1, 5)
print linear([0., 0., 0., 0., 1.]).astype("float32")

#- Print function
theano.printing.debugprint(linear)

#- Advanced :: You can evaluate symbolic graph without compilation
print c.eval({
                        a : np.int32(16), 
                        b : np.float32(12.10)
                     })

#=============

x = T.fvector("x")

### Basic Math operation & Activation funcsions
exp_x = T.exp(x)
sigmoid_x = T.nnet.sigmoid(x)
tanh_x = T.tanh(x)

### Advanced:: condition and comparison
#max(0,x)
relu_x = T.switch(T.gt(x, 0), x, 0)

f = theano.function(inputs=[x], outputs=[exp_x, sigmoid_x, tanh_x, relu_x])
f(np.array([-2., -1., 1., 2., 3.]).astype("float32"))

#===============

# y = x ** 2
x = T.fscalar("x")
y = x ** 2
gy = theano.grad(cost=y, wrt=x) ## 2x

f = theano.function(inputs=[x], outputs=[y, gy]) ## x**2, 2x
print f(10)

#==================

##Define a function which update t by 1 for each call.
t = theano.shared(np.int32(0))
increment = theano.function(inputs=[], outputs=t, updates=OrderedDict({t: t+1}) ) #OrderedDict({before update: after update})
for i in xrange(5):
    t = increment()
    print t

#===============

# Linear Regression
rng = np.random.RandomState(1234)

##  Step1. Define Symbolic / Shared Variables
x, t = T.fvector("x"), T.fvector("t") #inputs


W = theano.shared(rng.uniform(low=-0.08,high=0.08, size=(5, 3)).astype('float32'), name="W") #variables that are shared over iterations
b =  theano.shared(np.zeros(3).astype('float32'), name="b")


## Step2. Define graph
#y = T.dot(x, W) + b
y = T.nnet.sigmoid(T.dot(x, W) + b)
#y = T.tanh(T.dot(x, W) + b)
cost = T.sum((y - t)**2)


gW, gb = T.grad(cost=cost, wrt=[W, b]) # Take gradient

updates =  OrderedDict({W: W-0.01*gW, b: b-0.01*gb}) # Set update expression in OrderedDict


## Step3. Compile graph
f = theano.function(inputs=[x, t], outputs=[cost, gW, gb], updates=updates, allow_input_downcast=True)

## Step4. Run!!
for epoch in xrange(5):
    cost, gW, gb = f([-2., -1., 1., 2., 3.], [.4, .3, .5])
    print "epoch:: %d, cost:: %.3f"%(epoch, cost)

#============

train_y = np.eye(10)[train_y]
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

#--- Multi Layer Perceptron
class Layer:
    #- Constructor
    def __init__(self, in_dim, out_dim, function):
        self.in_dim   = in_dim
        self.out_dim  = out_dim                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        self.function = function
        self.W        = # WRITE ME!
        self.b        = # WRITE ME!
        self.params   = [# WRITE ME!]

    #- Forward Propagation
    def f_prop(self, x):
        self.z = # WRITE ME!
        return self.z

#--- Stochastic Gradient Descent
def sgd(params, g_params, eps=np.float32(0.1)):
    updates = OrderedDict()
    for param, g_param in zip(params, g_params):
        # WRITE ME!
    return updates

layers = [
    Layer(10,10),
    Layer(10,10)
    # WRITE ME!
]

x = T.fmatrix('x')
t = T.imatrix('t')

params = []
for i, layer in enumerate(layers):
    params += layer.params
    if i == 0:
        layer_out = layer.f_prop(x)
    else:
        layer_out = layer.f_prop(layer_out)

y = layers[-1].z
cost = T.mean(T.nnet.categorical_crossentropy(y, t))

g_params = T.grad(cost=cost, wrt=params)
updates = sgd(params, g_params)

train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

batch_size = 100
n_batches = train_X.shape[0]//batch_size
for epoch in xrange(5):
    train_X, train_y = shuffle(train_X, train_y)
    for i in xrange(n_batches):
        start = i*batch_size
        end = start + batch_size
        train(train_X[start:end], train_y[start:end])
    valid_cost, pred_y = valid(valid_X, valid_y)
    print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))
