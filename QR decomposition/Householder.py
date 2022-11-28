import numpy as np
from scipy.linalg import block_diag



def QR_hdec(A):
    n = np.shape(A)[0]
    p = np.shape(A)[1]
    e = np.concatenate((np.array([1]), np.array([0] * (n - 1))))
    Q = np.identity(n)
    t = min(n - 1, p)
    X = A

    for i in range(1, t + 1):
        x = X[:, 0]
        a = -np.sign(x[0]) * np.linalg.norm(x)
        e = np.concatenate((np.array([1]), [0] * (np.shape(x)[0] - 1)))
        u = x - a * e
        v = u / np.linalg.norm(u)
        Q = np.dot(Q, block_diag(np.identity(i-1), np.identity(len(v))-2*np.outer(v, v)))

        X = np.dot(Q, A)
        X = np.delete(X, obj = range(0, i), axis = 0)
        X = np.delete(X, obj = range(0, i), axis = 1)

    R = np.dot(np.matrix.transpose(Q), A)

    return Q, R

# Wikipedia example

A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])

temp = QR_hdec(A)
Q = temp[0]
R = temp[1]

print(Q)
print(R)
print(np.dot(np.matrix.transpose(Q), Q))
print(A-np.dot(Q, R))

# Matrix with more columns than rows

A = np.array([[12, -51, 4, 1], [6, 167, -68, 2], [-4, 24, -41, 3]])

temp = QR_hdec(np.matrix.transpose(A))
Q = temp[0]
R = temp[1]

print(Q)
print(R)
print(np.dot(np.matrix.transpose(Q), Q))
print(A-np.dot(np.matrix.transpose(R), np.matrix.transpose(Q)))