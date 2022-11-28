import numpy as np
import math
import matplotlib.pyplot as plt

def model_function(x, beta):
    n = len(x)
    p = len(beta)
    h = np.zeros(n)

    for i in range(n):
        for j in range(p):
            h[i] = h[i] + math.cos((j + 1) * beta[j] * x[i])

    return h


def jacobian(x, beta):
    n = len(x)
    p = len(beta)
    J = np.empty(shape=(n, p))

    for i in range(n):
        for j in range(p):
            J[i, j] = -(i + 1) * x[i] * math.sin((i + 1) * beta[j] * x[i])

    return J

n = 20 # number of samples
p = 30 # number of variables


e = np.random.normal(loc= 0, scale= 0.01, size= n) # noise
x = np.random.normal(loc= 0, scale= 1, size= n) # samples
beta_true = np.random.normal(loc=0, scale= 0.5, size= p) # covariates
z = model_function(x, beta_true) + e # observations


beta = beta_true + np.random.normal(loc=0, scale= 0.1, size= p) # starting point of the algorithm
lambda_param = 2 * pow(10, -9)
max_iter = 300
Delta = []


for i in range(max_iter):
    beta_old = beta

    J = jacobian(x, beta)
    y = z - model_function(x, beta) + J @ beta
    beta = np.linalg.inv(np.transpose(J) @ J + np.diag(np.repeat(lambda_param, p))) @ np.transpose(J) @ y

    Delta.append(np.linalg.norm(beta_old - beta))


x = list(range(len(Delta)))
y = Delta
plt.scatter(x, y)
plt.show()