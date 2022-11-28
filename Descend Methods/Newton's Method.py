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


x = np.random.normal(loc=0, scale=1, size=2)
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
    H_inv = np.linalg.inv(H)
    y = x - t * H_inv @ G
    lmbd = np.transpose(G) @ H_inv @ G
    while (modelFunction(y) > modelFunction(x) - a * t * lmbd):
        t = b * t
        y = x - t * H_inv @ G

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


'''
This example is from the book of Boyd & Vanderberghe - CUP 2004, pg. 492
'''
