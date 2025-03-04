import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
    
sigmoid_values = sigmoid(x_values)
tanh_values = tanh(x_values)
relu_values = relu(x_values)

plt.figure(figsize=(10, 6))

plt.plot(x_values, sigmoid_values, label="Sigmoid", linewidth=2)
plt.plot(x_values, tanh_values, label="Tanh", linewidth=2)
plt.plot(x_values, relu_values, label="ReLU", linewidth=2)

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.legend()
plt.title("Comparison of Activation Functions")
plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.grid(True)
plt.show()