import numpy as np

array = np.array([7, 14, 21])
new_array = array[:, np.newaxis]
print(new_array)
print(new_array.ndim)