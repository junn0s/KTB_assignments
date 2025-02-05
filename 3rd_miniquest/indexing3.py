import numpy as np

array = np.array([5, 15, 8, 20, 3, 12])
index = np.where(array > 10)[0]
print(index)