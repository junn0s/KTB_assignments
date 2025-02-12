import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.rand(50) * 10 
y = np.random.rand(50) * 10

plt.scatter(x, y, color='skyblue', marker='^', alpha=0.7)
plt.show()