import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(500)

sns.histplot(data, bins=30, kde=True, color='skyblue')
plt.title('gaussian distribution')
plt.show()