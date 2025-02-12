import numpy as np
import matplotlib.pyplot as plt

data1 = np.random.randn(10000)
data2 = np.random.randn(10000) + 3
plt.hist(data1, bins=50, color='skyblue', alpha=0.5, label='group1')
plt.hist(data2, bins=50, color='lightpink', alpha=0.5, label='group2')
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('gaussian distribution graph')
plt.show()