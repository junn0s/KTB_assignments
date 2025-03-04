import numpy as np

y_pred = np.array([2.5, 0.0, 2.1, 1.5])
y_true = np.array([3.0, -0.5, 2.0, 1.0])

def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

mse = mean_squared_error(y_true, y_pred)
print("평균 제곱 오차 (MSE):", mse)