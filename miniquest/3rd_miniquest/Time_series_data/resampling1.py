import pandas as pd
import numpy as np

date_rng = pd.date_range(start="2024-01-01", end="2024-01-05", freq="3h")

df = pd.DataFrame({
    "datetime": date_rng,
    "value": np.random.randint(10, 100, size=len(date_rng))  
})

df.set_index("datetime", inplace=True)
# 하루 단위로 다운샘플링 (평균 값 사용)
df_daily = df.resample("D").mean()  # 'D'는 하루 단위 리샘플링을 의미

print(df_daily)