import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 시계열 데이터 생성
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", periods=100, freq="D")  # 100일간의 날짜 생성
values = np.cumsum(np.random.randn(100))

df = pd.DataFrame({"Date": date_range, "Value": values})  # 날짜와 값을 포함한 데이터프레임 생성

# 선 그래프 시각화
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
sns.lineplot(x="Date", y="Value", data=df, color="blue", marker="o")  # 시계열 선 그래프
plt.xlabel("Date")  # X축 라벨 설정
plt.ylabel("Value")  # Y축 라벨 설정
plt.title("Time Series Line Plot")  # 그래프 제목 설정
plt.xticks(rotation=45)  # X축 눈금 회전
plt.show()  # 그래프 출력