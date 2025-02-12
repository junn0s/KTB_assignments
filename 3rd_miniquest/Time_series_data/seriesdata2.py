import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 시계열 데이터 생성
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", periods=100, freq="D")  # 100일간의 날짜 생성
values = np.cumsum(np.random.randn(100))

df = pd.DataFrame({"Date": date_range, "Value": values})  # 날짜와 값을 포함한 데이터프레임 생성

# 이동 평균(7일 평균) 계산
df["Moving_Avg"] = df["Value"].rolling(window=7).mean()  # 7일 이동 평균 계산

# 이동 평균 그래프 시각화
plt.figure(figsize=(10, 5))
sns.lineplot(x="Date", y="Value", data=df, label="Original Data", color="gray")  # 원본 데이터
sns.lineplot(x="Date", y="Moving_Avg", data=df, label="7-Day Moving Average", color="red")  # 이동 평균

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series with Moving Average")
plt.legend()  # 범례 추가
plt.xticks(rotation=45)
plt.show()