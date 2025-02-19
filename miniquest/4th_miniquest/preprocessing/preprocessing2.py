import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 가상의 데이터셋 생성
data = {
    '제품': ['A', 'B', 'C', 'D', 'E'],
    '가격': [100, 150, 200, 0, 250],
    '판매량': [30, 45, np.nan, 55, 60]
}

df = pd.DataFrame(data)

df.loc[:, '판매량'] = df['판매량'].fillna(df['판매량'].median())
df = df[df['가격'] > 0]
scaler = StandardScaler()
df[['가격', '판매량']] = scaler.fit_transform(df[['가격', '판매량']])

train, test = train_test_split(df, test_size=0.2, random_state=42)
print("\n학습용 데이터셋:")
print(train)
print("\n검증용 데이터셋:")
print(test)