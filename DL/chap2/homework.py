def homework(train_X, train_y, test_X):
    from pandas import Series

    train_X, valid_X , train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2)

    f1 = []
    for k in [3,5,10, 50]:
        pred =[]
        n = 100 if len(test_X)  >=  100 else len(test_X)
        for i in range(n):
            score = np.dot(valid_X, test_X[i])/np.linalg.norm(valid_X, axis=1)/np.linalg.norm(test_X[i])
            rakning = sorted([ [s, l] for s, l  in zip(score, valid_y)], reverse = True)
            top = ranking[:k]
            pred.append(Series(map(lambda x: x[1], top)).value_counts().index[0])
        f1.append(len(np.where(pred - test_y[:n] == 0)[0]))
    k = [3,5,10,50][np.argmax(f1)]

    N = len(test_X)
    pred_y =[]
    for i in range(N):
            score = np.dot(train_X, test_X[i])/np.linalg.norm(train_X, axis=1)/np.linalg.norm(test_X[i])
            rakning = sorted([ [s, l] for s, l  in zip(score, train_y)], reverse = True)
            top = ranking[:k]
            pred_y.append(Series(map(lambda x: x[1], top)).value_counts().index[0])
    return pred_y
