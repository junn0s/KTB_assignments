import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

date_rng = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
df = pd.DataFrame({
    "datetime": date_rng,
    "value": np.random.randint(50, 150, size=len(date_rng))
})

df.set_index("datetime", inplace=True)
df["SMA_7"] = df["value"].rolling(window=7).mean()
df_volatility = df[(df['value'] >= df['SMA_7'] * 1.2) | (df['value'] <= df['SMA_7'] * 0.8)]

print("변동성이 큰 날 (7일 SMA 대비 ±20% 이상 차이):")
print(df_volatility)