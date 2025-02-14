import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)
y = np.sin(x)

plt.figure(figsize=(10,5))
sns.lineplot(x=x, y=y, color='lightpink')
plt.show()