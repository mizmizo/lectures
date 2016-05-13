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

