import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix[:, 0]) # 열 출력
print(matrix[1, :]) # 행 출력