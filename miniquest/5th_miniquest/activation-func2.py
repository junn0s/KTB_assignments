import numpy as np

input_value = 0.5
weight = 0.8
bias = 0.1

weighted_sum = input_value * weight + bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
    

output_sigmoid = sigmoid(weighted_sum)
print(f"시그모이드: {output_sigmoid}")

output_tanh = tanh(weighted_sum)
print(f"하이퍼볼릭 탄젠트: {output_tanh}")

output_relu = relu(weighted_sum)
print(f"렐루: {output_relu}")