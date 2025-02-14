import pandas as pd

data = {
    'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'Close': [101, 104, 106, 105, 109, 108, 111, 113, 116, 119]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

print("5일 이동평균과 지수 이동평균:")
print(df)