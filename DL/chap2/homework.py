def homework(train_X, train_y, test_X):
    from pandas import Series

    train_X, valid_X , train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1)

    f1 = []
    split_n = 10
    for k in [3,5,7,10,20,50]:
        pred =[]
        n = len(valid_X)
        for i in range(n):
            top = []
            for j in range(split_n):
                score = np.dot(train_X[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1], valid_X[i])/np.linalg.norm(train_X[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1], axis=1)/np.linalg.norm(valid_X[i])
                ranking = sorted([ [s, l] for s, l  in zip(score, train_y[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1])], reverse = True)
                top = sorted(top + ranking[:k], reverse = True)[:k]
            pred.append(Series(map(lambda x: x[1], top)).value_counts().index[0])
        f1.append(len(np.where(pred - valid_y == 0)[0]))
    k = [3,5,7,10,20,50][np.argmax(f1)]
    print k

    N = len(test_X)
    pred_y =[]
    for i in range(N):
        top = []
        for j in range(split_n):
            score = np.dot(train_X[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1], test_X[i])/np.linalg.norm(train_X[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1], axis=1)/np.linalg.norm(test_X[i])
            ranking = sorted([ [s, l] for s, l  in zip(score, train_y[int(N * j / split_n) : int((N * (j + 1) /split_n)) - 1])], reverse = True)
            top = sorted(top + ranking[:k], reverse = True)[:k]
        pred_y.append(Series(map(lambda x: x[1], top)).value_counts().index[0])
    return pred_y
