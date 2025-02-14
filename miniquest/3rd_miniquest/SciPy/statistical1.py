import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.normal(loc=70, scale=20, size=1000)

plt.figure(figsize=(8, 6))
plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("normal distribution box plot")
plt.ylabel("value")
plt.grid(True)
plt.show()