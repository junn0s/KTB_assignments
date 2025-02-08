import numpy as np

array = np.array([100, 200, 300])
new_array = array.astype(np.uint8)
print(new_array.nbytes)