import numpy as np

def Q2(A, B, C, x):
    return numpy.dot(A*x, B) - C.T

def Q3(x, a):
    rank = sorted([[s,l] for s,l in zip(np.linalg.norm(x-a, axis=1), x)])
    return np.array([row[1] for row in rank])

def Q4(M, N, x0, x1, y0, y1):
    MAP = np.arange(M*N).reshape(N,M)
    cut = MAP[y0:y1+1,x0:x1+1]
    return cut.sum()

def Q5_1(x):
    one_hot = np.zeros((len(x), x.max() + 1))
    one_hot[np.arange(len(x)), x] = 1
    return one_hot

def Q5_2(x):
    return np.where(x == 1)[1]


def Q6(m, n):
    a = np.empty((0,m), int)
    b = np.arange(1,m + 1)
    c = np.arange(1,n + 1)
    return np.array([np.append(a, np.array([b*i])) for i in c])

def Q7(x):
    y = x[1:len(x)]
    return y - x[:len(x) - 1]
