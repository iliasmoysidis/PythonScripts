#####################################################################################################
#########In this script we find a feasible point for the set of restrictions Cx<=d and Ax=b##########
#######################################A is nxp, with n<=p###########################################
#####################################################################################################
import math
import numpy as np

def barrierFunction(t, e, C, d, u):

    b = t * np.transpose(e) @ u

    for i in range(len(d)):
        a = np.reshape(C[i, :], (np.shape(C)[1], 1))
        b += math.log(d[i] - np.transpose(a) @ u)

    return b


def Gradient(t, e, C, d, u):

    g = t * e

    for i in range(len(d)):
        a = np.reshape(C[i, :], (np.shape(C)[1], 1))
        g += a / (d[i] - np.transpose(a) @ u)

    return g

def Hessian(C, d, u):
    H = np.zeros(shape = (np.shape(C)[1], np.shape(C)[1]))

    for i in range(len(d)):
        a = np.reshape(C[i, :], (np.shape(C)[1], 1))
        H += np.outer(a = a, b = a) / (d[i] - np.transpose(a) @ u)

    return H


# Problem dimensions
n = 3
m = 4
p = 5
###########################################
###################Step 1##################
##########Find a solution for Ax=b#########
###########################################



A = np.reshape(np.random.normal(loc = 0, scale = 1, size = n * p), (n, p))
C = np.reshape(np.random.normal(loc = 0, scale = 1, size = m * p), (m, p))
b = np.reshape(np.random.normal(loc = 0, scale = 1, size = n), (n, 1))
d = np.reshape(np.random.normal(loc = 0, scale = 1, size = m), (m, 1))
##############################################
############Problem Reformulation#############
##############################################
tildeA = np.concatenate((A, np.zeros(shape = (n, 1))), axis = 1)
tildeC = np.concatenate((C, -np.ones(shape = (m, 1))), axis = 1)
e = np.concatenate((np.zeros(shape = (p, 1)), [[1]]), axis= 0)




####################################
###########Initial values###########
####################################
Q, R = np.linalg.qr(np.transpose(A))
x = Q @ np.linalg.inv(np.transpose(R)) @ b # a solution of Ax = b, pg. 681 Boyd & Vanderberghe CUP 2004
s = np.max(C @ x - d) + 1 # so that d-Cx+s>0
u = np.concatenate((x, [[s]]), axis = 0)



maxIter = 30
a = 0.1
r = 0.7
t = 1 #step size for barrier method
mu = 2 # increase of step size at each iteration for barrier method

conditionOuter = True
i = 0
while (conditionOuter):
    u[-1] = s

    conditionInner = True
    j = 0
    while (conditionInner):
        # calculating Newton step Du for variable u
        G = Gradient(t, e, tildeC, d, u)
        H = Hessian(tildeC, d, u)
        arr1 = np.concatenate((H, np.transpose(tildeA)), axis=1)
        arr2 = np.concatenate((tildeA, np.zeros(shape=(np.shape(tildeA)[0], np.shape(tildeA)[0]))), axis=1)
        D = np.concatenate((arr1, arr2), axis=0)
        D_inv = np.linalg.inv(D)
        J = np.concatenate((-G, np.zeros(shape=(n, 1))), axis=0)
        Du = (D_inv @ J)[0:(p + 1)]

        # backtracking line search
        q = 1
        y = u + q * Du
        while (np.any(d-tildeC @ y <= 0)): # making sure new point is inside domain
            q = r * q
            y = u + q * Du
        while (barrierFunction(t, e, tildeC, d, y) > barrierFunction(t, e, tildeC, d, u) + a * q * np.transpose(
                G) @ Du):
            q = r * q
            y = u + q * Du
            while (np.any(d - tildeC @ y <= 0)): # making sure new point is inside domain
                q = r * q
                y = u + q * Du

        u = y
        j += 1
        conditionInner = (u[-1] > 0 and j < maxIter)

    t = mu * t
    i += 1
    conditionOuter = (u[-1] > 0 and i < maxIter)
    s = np.max(C @ u[0:p] - d) + 1


s = u[-1]
x = u[0:p]
print(s) # if negative it is a feasible initial point, if positive the domain is empty
print(C @ x - d) # if all negative we found a point that satisfies the inequalities