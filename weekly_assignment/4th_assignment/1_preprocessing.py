# 가상의 학생 성적 데이터를 활용한 데이터 전처리 과정

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 학생 성적 데이터
data = {
    '학생': ['milo', 'noah', 'dobby', 'ethan', 'sandy', 'sophie'],
    '수학': [90, np.nan, 85, 88, np.nan, 86],
    '영어': [80, 78, np.nan, 90, 85, np.nan],
    '과학': [np.nan, 89, 85, np.nan, 80, 90]
}

# 전처리 - 결측치 처리
df = pd.DataFrame(data)
df.loc[:, '수학'] = df['수학'].fillna(df['수학'].mean())
df.loc[:, '영어'] = df['영어'].fillna(df['영어'].mean())
df.loc[:, '과학'] = df['과학'].fillna(df['과학'].mean())

# 전처리 - 이상치 제거
df = df[(df['수학'] >= 0) & (df['수학'] <= 100)]
df = df[(df['영어'] >= 0) & (df['영어'] <= 100)]
df = df[(df['과학'] >= 0) & (df['과학'] <= 100)]

# 전처리 - 데이터 정규화
scaler = MinMaxScaler()  # 0과 1 사이로 변환
df[['수학', '영어', '과학']] = scaler.fit_transform(df[['수학', '영어', '과학']])

# 전처리 - 데이터 분할
train, test = train_test_split(df, test_size=0.2)
print("\n학습용 데이터셋:")
print(train)
print("\n검증용 데이터셋:")
print(test)