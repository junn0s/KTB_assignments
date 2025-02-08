import pandas as pd
import numpy as np

data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
        '나이': [25, np.nan, 30, 28],
        '성별': ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)
print(df.isnull())
print(df.isnull().sum())