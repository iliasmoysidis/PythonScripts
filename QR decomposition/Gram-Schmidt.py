import numpy as np


def proj(a, b):
    d = np.dot(a, b)*b/np.dot(b, b)
    return d


def QR_dec(A):

    n = np.shape(A)[0]
    p = np.shape(A)[1]

    U = np.empty(shape=(n, p))
    Q = np.empty(shape=(n, p))

    U[:, 0] = A[:, 0]
    Q[:, 0] = U[:, 0] / np.linalg.norm(U[:, 0])

    for j in range(1, p):
        U[:, j] = A[:, j]

        for i in range(0, j):
            U[:, j] -= proj(A[:, j], U[:, i])

        Q[:, j] = U[:, j] / np.linalg.norm(U[:, j])

    R = np.dot(np.matrix.transpose(Q), A)

    return Q, R










# Wikepedia example

A = np.array([[12, -51, 4], [6,167,-68], [-4, 24, -41]])

temp = QR_dec(A)
Q = temp[0]
R = temp[1]


print(Q)
print(R)