############################################################################
##########Quadratic Program with only inequality constraints################
############################################################################

import numpy as np
import matplotlib.pyplot as plt

def obj_const(A, b, x):
    n = A.shape[0]
    c = np.zeros(shape=n)
    for i in range(n):
        c[i] = b[i] - A[i, :] @ x
    return c

#####################################################################
#######################Problem Formulation###########################
#####################################################################


# minimize 1/2 x'Hx + f'x
# subject to Ax<=b

#########################################
###########Problem Dimensions############
#########################################

# Define dimensions of problem
n = 200 # number of inequalities
p = 150 # number of variables

# p is the number of variables and n is the number of inequalities.
# For this method to work we need p <= n

# Construct H
X = np.reshape(np.random.normal(loc=0, scale=1, size=p**2), (p, p))
H = np.transpose(X) @ X

# Construct f
f = np.random.normal(loc=0, scale=1, size=p)

# Construct A, b
A = np.reshape(np.random.normal(loc=0, scale=1, size=n*p), (n, p))
b = np.random.normal(loc=0, scale=1, size=n)


###################################################################
########################Barrier Method#############################
###################################################################


max_iter = 30 # maximum number of iterations
mu = 1 # penalty parameter of barrier function
x = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ b # initial value of x. must be feasible
lmbd = np.random.exponential(scale=1, size=n) # initial value of dual variable

Delta = []

for i in range(max_iter):
    x_old = x
    c = obj_const(A, b, x)

    # Construct G
    arr1 = H @ x + f + np.transpose(A) @ lmbd
    arr2 = np.diag(lmbd) @ c - mu * np.ones(n)
    G = np.concatenate((arr1, arr2), axis=0)

    # Construct the jacobian of G
    arr3 = np.concatenate((H, np.transpose(A)), axis=1)
    arr4 = np.concatenate((-np.diag(lmbd) @ A, np.diag(c)), axis=1)
    J = np.concatenate((arr3, arr4), axis=0)
    J_inv = np.linalg.inv(J)

    # Newton's updates
    temp = np.concatenate((x, lmbd), axis=0) - J_inv @ G

    # Make sure the lambdas are all positive
    a = 1
    while(sum(temp[p:(p + n)] > 0) < n):
        a = 0.9 * a
        temp = np.concatenate((x, lmbd), axis=0) - a * J_inv @ G

    x = temp[0:p]
    lmbd = temp[p:(p + n)]
    mu = 0.9 * mu # reduce the penalty parameter at each step
    Delta.append(np.linalg.norm(x_old - x))


x = list(range(len(Delta)))
y = Delta
plt.scatter(x, y)
plt.show()