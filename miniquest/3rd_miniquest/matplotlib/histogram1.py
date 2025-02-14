import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)
plt.hist(data, bins=15, color='skyblue', edgecolor='black')
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('gaussian distribution graph')
plt.show()