import numpy as np
import matplotlib.pyplot as plt

normal_data = np.random.randn(1000)  
uniform_data = np.random.rand(1000)  

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

axes[0].hist(normal_data, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title("Histogram of Normal Distribution")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(uniform_data, bins=30, color='lightgreen', edgecolor='black')
axes[1].set_title("Histogram of Uniform Distribution")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()