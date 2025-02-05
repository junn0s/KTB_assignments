import numpy as np

array = np.array([[3, 7, 2], [8, 4, 6]])

# 축 0 
max_axis0 = np.max(array, axis=0)
print(max_axis0)      
print(max_axis0.ndim)

# 축 1 
max_axis1 = np.max(array, axis=1)
print(max_axis1)       
print(max_axis1.ndim)  