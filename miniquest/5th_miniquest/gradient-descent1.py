import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta = np.random.randn(2, 1)

def gradient_descent(X_b, y, theta, learning_rate, n_iterations):
    m = len(X_b)  # 데이터 개수
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

X_b = np.c_[np.ones((100, 1)), X]  # X 앞에 1열을 추가해 절편 항을 포함

learning_rate = 0.1
n_iterations = 1000

theta_best = gradient_descent(X_b, y, theta, learning_rate, n_iterations)
print("최적의 파라미터:", theta_best)