import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)
x = np.random.randn(50) * 2
y = np.random.randn(50) * 2
categories = np.random.choice(['A', 'B', 'C'], size=50)
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}

for cat in np.unique(categories): 
    idx = categories == cat  
    plt.scatter(x[idx], y[idx], 
                color=colors[cat],
                label=f'Category {cat}',  
                alpha=0.7, 
                s=80)  

plt.xlabel("X")  
plt.ylabel("Y")  
plt.title("Scatter Plot by Category")  
plt.legend()  
plt.show()  