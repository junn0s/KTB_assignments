import numpy as np

array = np.array([1, np.e, 10, 100])
log_array = np.log(array)

result = log_array[log_array > 1]
print(result)