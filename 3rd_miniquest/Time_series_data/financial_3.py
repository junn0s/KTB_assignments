import pandas as pd
import numpy as np

date_rng = pd.date_range(start='2024-01-01', periods=30, freq='D')
close_prices = np.random.uniform(100, 200, size=len(date_rng))

df = pd.DataFrame({'Close': close_prices}, index=date_rng)

weekly_close_mean = df['Close'].resample('7D').mean()
weekly_volatility = df['Close'].resample('7D').std()

print("주간 종가 평균:")
print(weekly_close_mean)
print("\n주간 변동성 (표준편차):")
print(weekly_volatility)