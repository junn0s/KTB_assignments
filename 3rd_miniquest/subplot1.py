import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y1 = x ** 2  
y2 = x ** 3 

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

axes[0].plot(x, y1, color='skyblue')
axes[0].set_xlabel("X-axis") 
axes[0].set_ylabel("Y-axis")
axes[0].legend()

axes[1].plot(x, y2, color='lightpink')
axes[1].set_xlabel("X-axis") 
axes[1].set_ylabel("Y-axis")
axes[1].legend()

plt.tight_layout()
plt.show()