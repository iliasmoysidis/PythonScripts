import numpy as np
import random
import numpy.random
import matplotlib.pyplot as plt

def value_function_estimator(R):

   t = len(R)
   k = len(R[0])
   Q = np.zeros(k, dtype=object)

   for j in range(k):
      x = 0
      y = 0
      for i in range(t):
         if R[i, j] != 0:
            x += R[i, j]
            y += 1

         if y != 0:
            Q[j] = x/y

   return Q

def argmax(Q):
   max_element = max(Q)
   index_list = np.where(Q == max_element)[0]
   a = random.choice(index_list)

   return a



n = 1000 # number of iterations
k = 10 # number of slots
epsilon = 0.3 # this is a number between (0,1) that promotes exploration
R = np.zeros(shape=(n, k)) # reward matrix
Q = np.zeros(shape=(n, k)) # value estimator matrix

q = np.random.normal(loc = 0, scale = 1, size = k) # generate true values


for i in range(n):
   if i == 0: # randomly pick first slot
      a = random.randint(0, 9)  # randomly select one of the k levers
      R[i, a] = numpy.random.normal(loc=q[a], scale=1, size=1) # generate reward with standard normal noise
   else:
      if epsilon>0:
         Q[i, :] = value_function_estimator(R[0:i, :])
         x = np.random.uniform(low=0, high=1)
         if x < epsilon: # explore with probability epsilon
            a = random.randint(0, 9)
            R[i, a] = numpy.random.normal(loc=q[a], scale=1, size=1)
         else:
            a = argmax(Q[i,:]) # find the best action
            R[i, a] = np.random.normal(loc=q[a], scale=1, size=1)


# Plot convergence of algorithm
x = list(range(n))
y = np.amax(Q, axis=1)
plt.scatter(x, y)
plt.show()