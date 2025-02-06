import pandas as pd

data = {'이름': ['홍길동', '김철수', '박영희'], 
        '나이': [25, 30, 28], 
        '성별': ['남', '남', '여']}

df = pd.DataFrame(data)
df = df.sort_values(by='나이')
print(df)