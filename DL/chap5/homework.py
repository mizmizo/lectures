def homework(train_X, test_X, train_y):
    np.seterr(divide='ignore', invalid='ignore', over = 'ignore')
    layer_num = 2
    out_dim = 10
    eps = 1.0
    dims = [len(train_X[0]),40,out_dim]

    wname = "w" + str(1)
    w = [theano.shared(np.random.uniform(low=-0.08, high=0.08, size=(dims[0], dims[1])).astype("float32"), name=wname)]
#     w = [np.random.uniform(low = -0.08, high = 0.08, size = (dims[0],dims[1])).astype("float64")]
    bname = "b" + str(1)
    b = [theano.shared(np.zeros(dims[1]).astype("float32"), name=bname)]
#     b = [np.zeros(dims[1]).astype("float64")]
    uname = "u" + str(1)
    u = [theano.shared(np.zeros(dims[1]).astype("float32"), name=uname)]
#    u = [np.zeros(dims[1]).astype("float64")]
    zname = "z" + str(1)
    z = [theano.shared(np.zeros(dims[1]).astype("float32"), name=zname)]
#    z = [np.zeros(dims[1]).astype("float64")]
    dname = "delta" + str(1)
    delta = [theano.shared(np.zeros(dims[1]).astype("float32"), name=dname)]
#    delta = [np.zeros(dims[1]).astype("float64")]

    for i in range(1, layer_num):
    wname = "w" + str(i)
    w.append(theano.shared(np.random.uniform(low=-0.08, high=0.08, size=(dims[i], dims[i + 1])).astype("float32"), name=wname))
    bname = "b" + str(i)
    b.append(theano.shared(np.zeros(dims[i + 1]).astype("float32"), name=bname))
    uname = "u" + str(i)
    u.append(theano.shared(np.zeros(dims[i + 1]).astype("float32"), name=uname))
    zname = "z" + str(i)
    z.append(theano.shared(np.zeros(dims[i + 1]).astype("float32"), name=zname))
    dname = "delta" + str(i)
    delta.append(theano.shared(np.zeros(dims[i + 1]).astype("float32"), name=dname))

    z = T.fvector("z")
    w = T.fmatrix("w")
    b = T.fvector("b")
    u = T.dot(z, w) + b
    u_prop = theano.function(inputs = [z, w, b], outputs = u)
    z = T.nnet.sigmoid(u)
    z_prop = theano.function(inputs = [u], outputs = z)

    for epoch in xrange(10):
        for x, y in zip(train_X, train_y):
            tmp_z = x[np.newaxis, :]
            y2 = np.zeros((1,10))
            y2[0,y] = 1

            #f_prop
            for i in xrange(layer_num - 1):
                if i == 0:
                    u[i] = u_prop(tmp_z, w[i], b[i])
#                    u[i] = np.dot(tmp_z, w[i]) + b[i]
                else:
                    u[i] = u_prop(z[i - 1], w[i], b[i])
#                    u[i] = np.dot(z[i - 1], w[i]) + b[i]
                u[i][np.where(u < -20)] = -20
                z[i] = z_prop(u[i])
#                 z[i] = 1 / (1 + np.exp(-u[i])) #sigmoid 
            u[layer_num - 1] = u_prop(z[layer_num - 2], w[layer_num - 1], b[layer_num - 1])
#            u[layer_num - 1] = np.dot(z[layer_num - 2], w[layer_num - 1]) + b[layer_num - 1]
            u[layer_num - 1][np.where(u < -20)] = -20
#            z[layer_num - 1] = np.exp(u[layer_num -1]) / np.sum(np.exp(u[layer_num - 1]), axis = 1, keepdims = True) #softmax
            #cost = np.sum(-y2 * np.log(z[layer_num - 1]) - (1 - y2)*np.log(1 - z[layer_num - 1]))
            z[layer_num - 1] = T.nnet.softmax(u[layer_num - 1])
            tmp_delta = z[layer_num - 1] - y2

            #b_prop
            for i in xrange(layer_num):
                if i == 0:
                    delta[layer_num - 1] = tmp_delta
                else:
                    delta[layer_num - 1 - i] = np.dot(delta[layer_num - i], w[layer_num - i].T) * (1 / ( 1 + np.exp(u[layer_num - 1 - i]))) * (1 - (1 / ( 1 + np.exp(u[layer_num - 1 - i])))) #deriv_sigmoid
            #update parameter
            for i in xrange(layer_num):
                if i == 0:
                    dW = np.dot(tmp_z.T, delta[i])
                    db = np.dot(np.ones(len(tmp_z)), delta[i])
                else:
                    dW = np.dot(z[i - 1].T, delta[i])
                    db = np.dot(np.ones(len(z[i -1])), delta[i])
                w[i] = w[i] - eps * dW
                b[i] = b[i] - eps * db
    #test
    pred_y = np.zeros(len(test_X)).astype("int32")
    for j in xrange(len(test_X)):
        tmp_z = test_X[j][np.newaxis, :]
        for i in xrange(layer_num):
            if i == 0:
                u[i] = np.dot(tmp_z, w[i]) + b[i]
            else:
                u[i] = np.dot(z[i - 1], w[i]) + b[i]
            z[i] = 1 / (1 + np.exp(-u[i])) #sigmoid
        u[layer_num - 1] = np.dot(z[layer_num - 2], w[layer_num - 1]) + b[layer_num - 1]
        z[layer_num - 1] = np.exp(u[layer_num -1]) / np.sum(np.exp(u[layer_num - 1]), axis = 1, keepdims = True) #softmax
        pred_y[j] = np.argmax(z[layer_num - 1])

    return pred_y
