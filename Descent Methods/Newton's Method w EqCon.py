import numpy as np
import math
import matplotlib.pyplot as plt


def modelFunction(x):
    f = math.exp(x[0] + 3 * x[1] - 0.1) + math.exp(x[0] - 3 * x[1] - 0.1) + math.exp(-x[0] - 0.1)

    return f

def gradient(x):
    a = math.exp(x[0] + 3 * x[1] - 0.1) + math.exp(x[0] - 3 * x[1] - 0.1) - math.exp(-x[0] - 0.1)
    b = 3 * math.exp(x[0] + 3 * x[1] - 0.1) - 3 * math.exp(x[0] - 3 * x[1] - 0.1)
    F = np.array([a, b])

    return F

def hessian(x):
    a = x[0] + 3 * x[1] - 0.1
    b = x[0] - 3 * x[1] - 0.1
    c = -x[0] - 0.1

    A = np.zeros(shape=(2, 2))
    A[0, 0] = math.exp(a) + math.exp(b) + math.exp(c)
    A[0, 1] = 3 * (math.exp(a)-math.exp(b))
    A[1, 0] = A[0, 1]
    A[1, 1] = 9 * (math.exp(a) + math.exp(b))

    return A


###############################################################################
##############Newton's Method with backtracking line search####################
###############################################################################

# construct equality constraints Cx=d
C = np.reshape([2, 1], newshape=(1, 2))
d = 2

x = np.array([1, 0]) # solution of under-determined system
Delta = []
Theta = []
a = 0.1
b = 0.7
maxIter = 40


for i in range(maxIter):
    Theta.append(modelFunction(x))

    t = 1
    G = gradient(x)
    H = hessian(x)
    arr1 = np.concatenate((H, np.transpose(C)), axis=1)
    arr2 = np.concatenate((C, np.zeros(shape=(1, 1))), axis=1)
    R = np.concatenate((arr1, arr2), axis=0)
    Dx = (np.linalg.inv(R) @ np.concatenate((-G, np.zeros(1))))[0:2]
    y = x + t * Dx
    while (modelFunction(y) > modelFunction(x) + a * t * np.transpose(G) @ Dx):
        t = b * t
        y = x + t * Dx
    Delta.append(np.linalg.norm(x - y))
    x = y

# Convergence of argument
x = list(range(len(Delta)))
y = Delta
plt.scatter(x, y)
plt.show()

# Convergence of the value of the objective function
x = list(range(len(Theta)))
y = Theta
plt.scatter(x, y)
plt.show()