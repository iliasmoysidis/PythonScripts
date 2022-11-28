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


###############################################################################
#############Gradient Descent with backtracking line search####################
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
    y = x - t * G
    while (modelFunction(y) > modelFunction(x) - a * t * np.linalg.norm(gradient(x))):
        t = b * t
        y = x - t * G
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
This example is from the book of Boyd & Vanderberghe - CUP 2004, pg. 470
'''