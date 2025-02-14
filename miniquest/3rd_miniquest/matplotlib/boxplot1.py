import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(50)

plt.boxplot(data)
plt.show()