import numpy as np

array1 = np.array([10, 20, 30])
array2 = np.array([1, 2, 3])
new_arr = np.empty_like(array1)

np.add(array1, array2, out=new_arr)
print(new_arr)