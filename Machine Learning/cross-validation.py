import numpy as np
import matplotlib.pyplot as plt

n = 100
p = 50
X = np.array(np.random.normal(100, 20, n * p)).reshape((n, p))
beta = np.array(np.random.normal(1, 3, p)).reshape((p, 1))
epsilon = np.array(np.random.normal(0, 1, n)).reshape((n, 1))
y = X @ beta + epsilon


lambda_seq = np.linspace(0.001, 10, 200)
error = []

idx_train = np.sort(np.random.choice(n, size=(round(0.9 * n)), replace=False))
idx_test = np.array(list(set(range(n)) - set(idx_train)))

X_train = X[idx_train, :]
y_train = y[idx_train, :]
X_test = X[idx_test, :]
y_test = y[idx_test, :]
for pen_param in lambda_seq:
    beta_lambda_hat = np.linalg.inv(np.transpose(X_train) @ X_train + pen_param * np.eye(p))\
                      @ np.transpose(X_train) @ y_train

    error.append(np.linalg.norm(y_test - X_test @ beta_lambda_hat))


min_pos = error.index(min(error))
lambda_hat = lambda_seq[min_pos]
beta_hat = np.linalg.inv(np.transpose(X_train) @ X_train + lambda_hat * np.eye(p))\
                      @ np.transpose(X_train) @ y_train

x = list(range(len(error)))
y = error
plt.scatter(x, y)
plt.show()




