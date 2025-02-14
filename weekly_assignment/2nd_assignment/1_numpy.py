import numpy as np

array1 = np.arange(1, 10).reshape(3, 3)
array2 = np.random.rand(2, 5)

sum1 = array1.sum()
avg1 = array1.mean()

sum2 = array2.sum()
avg2 = array2.mean()

# 1차원으로 평탄화 후 곱
new_arr1 = array1.flatten()
new_arr2 = array2.flatten()
min_length = min(new_arr1.size, new_arr2.size)  # 길이 9 기준으로 곱
result1 = new_arr1[:min_length] * new_arr2[:min_length]

# 브로드캐스팅(=외적) 곱
result2 = np.multiply.outer(array1, array2)

print()
print("1번 배열 합계 및 평균")
print("합계 : ", sum1)
print("평균 : ", avg1)
print("\n--------------------------------------\n")

print("2번 배열 합계 및 평균")
print("합계 : ", sum2)
print("평균 : ", avg2)
print("\n--------------------------------------\n")

print("1차원으로 평탄화 후 곱")
print(result1)
print("\n--------------------------------------\n")

print("브로드캐스팅(외적) 곱")
print(result2)
print("\n--------------------------------------\n")