import pandas as pd
import numpy as np

date_rng = pd.date_range(start="2024-01-01", end="2024-01-03", freq="3h")

df = pd.DataFrame({
    "datetime": date_rng,
    "value": np.random.randint(10, 100, size=len(date_rng))  
})

df.set_index("datetime", inplace=True)

# 1시간 단위로 업샘플링 (보간 없이 NaN 유지)
df_hourly = df.resample("h").asfreq()  # 'h'는 1시간 단위 리샘플링을 의미
# 선형 보간 (Linear Interpolation)
df_linear = df_hourly.interpolate(method="linear")

print(df_linear.head().reset_index())