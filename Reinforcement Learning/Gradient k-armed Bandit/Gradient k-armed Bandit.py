import numpy as np
import matplotlib.pyplot as plt


def softmax(H):
    r = np.exp(H)
    s = np.sum(r)
    dist = r/s

    return dist

n = 100 # number of iterations
s = 0.4 # step-size
k = 10 # number of "arms" or slots


# Set initial values
q = np.random.normal(loc=4,scale=1,size=k) # true rewards for each action
H = np.zeros(k) # initial preference function
R = 0 # initial value for sum of rewards
actions = np.zeros(n)


for i in range(n):

    probs = softmax(H) # probability distribution for each action
    a = np.random.choice(k, p = probs) # sample from categorical distribution
    r = np.random.normal(loc=q[a], scale=1, size=1) # generate reward based on true reward
    R = R+r # sum up the rewards from each iteration
    Q = R/(i+1) # compute the average reward for the i-th iteration

    for j in range(k):
        if j == a:
            H[j] = H[j] + s * (r - Q) * (1 - probs[j])
        else:
            H[j] = H[j] - s * (r - Q) * probs[j]


    actions[i] = a


print(q)
x = list(range(n))
y = actions
plt.scatter(x, y)
plt.show()