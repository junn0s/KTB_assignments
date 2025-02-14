import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.random.normal(loc=60, scale=15, size=500)

print(np.mean(data))
print(np.std(data))
print(np.min(data))
print(np.max(data))