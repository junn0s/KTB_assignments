import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100) * 10  
y = 2 * x + np.random.randn(100) 

sns.regplot(x=x, y=y)
plt.show()