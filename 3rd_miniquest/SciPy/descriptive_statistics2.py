import numpy as np
import pandas as pd

# 샘플 데이터 생성
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)  # 평균 50, 표준편차 10인 정규 분포 데이터 생성
df = pd.DataFrame(data, columns=["value"])  # 데이터프레임 생성

# 사분위 범위(IQR) 계산
q1 = np.percentile(df["value"], 25)  # "value" 열의 1사분위수(Q1) 계산
q3 = np.percentile(df["value"], 75)  # "value" 열의 3사분위수(Q3) 계산
iqr = q3 - q1  # 사분위 범위(IQR) 계산

# 이상값 탐지 (IQR 기준)
lower_bound = q1 - 1.5 * iqr  # IQR을 이용하여 하한 경계값 계산
upper_bound = q3 + 1.5 * iqr  # IQR을 이용하여 상한 경계값 계산

# 이상값 제거
df_no_outliers = df[(df["value"] >= lower_bound) & (df["value"] <= upper_bound)]  # 값이 하한 또는 상한 경계 안에 있는 데이터만 선택

origin_mean = df["value"].mean()
new_mean = df_no_outliers["value"].mean()

print(origin_mean)
print(new_mean)