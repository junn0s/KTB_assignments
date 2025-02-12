import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 시계열 데이터 생성
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", periods=100, freq="D")  # 100일간의 날짜 생성
values = np.cumsum(np.random.randn(100))

df = pd.DataFrame({"Date": date_range, "Value": values})  # 날짜와 값을 포함한 데이터프레임 생성

# 이상치 탐지를 위한 사분위 범위(IQR) 계산
Q1 = df["Value"].quantile(0.25)  # 1사분위수
Q3 = df["Value"].quantile(0.75)  # 3사분위수
IQR = Q3 - Q1  # IQR 계산
lower_bound = Q1 - 1.5 * IQR  # 하한선
upper_bound = Q3 + 1.5 * IQR  # 상한선

# 이상치 여부 판별
df["Outlier"] = (df["Value"] < lower_bound) | (df["Value"] > upper_bound)  # 이상치 여부 (True/False)

# 이상치 데이터 필터링
outliers = df[df["Outlier"]]

# 이상치 시각화
plt.figure(figsize=(10, 5))
sns.lineplot(x="Date", y="Value", data=df, label="Original Data", color="blue")
sns.scatterplot(x="Date", y="Value", data=outliers, color="red", label="Outliers", s=100)  # 이상치 표시

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series with Outlier Detection")
plt.legend()
plt.xticks(rotation=45)
plt.show()