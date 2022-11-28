import numpy as np
import matplotlib.pyplot as plt


def jacob(beta, x):
    n = len(x)
    p = len(beta)
    J = np.empty(shape=(n, p))
    for i in range(n):
        J[i, :] = [-x[i] / (beta[1] + x[i]), beta[0] * x[i] / ((beta[1] + x[i])**2)]
    return J

def model_function(beta, x):
    n = len(x)
    A = np.empty(n)
    for i in range(n):
        A[i] = beta[0] * x[i] / (beta[1] + x[i])
    return A


n = 20 # number of samples
p = 2 # number of variables


e = np.random.normal(loc=0, scale=0.1, size=n) # generate noise
x = np.random.normal(loc=0, scale=1, size=n) # generate covariates

beta_true = [0.6, 1.5]

# generate the observed values
y = model_function(beta_true, x)+e



########################################
############Gauss-Newton################
########################################

beta = [0.4, 1.6] # initial coefficients
max_iter = 30
Delta = []
Theta = []

for i in range(max_iter):
    beta_old = beta

    J = jacob(beta, x)
    r = y-model_function(beta, x)
    beta = beta - np.linalg.inv(np.transpose(J) @ J) @ np.transpose(J) @ r

    Delta.append(np.linalg.norm(beta_old - beta))
    Theta.append(np.linalg.norm(y-model_function(beta, x))**2)

print(beta)
print(Delta)
print(Theta)

x = list(range(len(Delta)))
y = Delta
plt.scatter(x, y)
plt.show()

# The algorithm depends heavily on the choice of initial values