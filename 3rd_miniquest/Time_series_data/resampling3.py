import pandas as pd
import numpy as np

date_rng = pd.date_range(start="2024-01-01", end="2024-01-07", freq="3h")

df = pd.DataFrame({
    "datetime": date_rng,
    "value": np.random.randint(10, 100, size=len(date_rng))  
})

df.set_index("datetime", inplace=True)

# 하루 단위로 다운샘플링, 최소 최대 구하기
daily_min = df.resample("D").min()
daily_max = df.resample("D").max()

# 결과 출력
print("각 날짜별 최소 값:")
print(daily_min)
print("\n각 날짜별 최대 값:")
print(daily_max)