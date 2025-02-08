import pandas as pd

df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [15000, 25000, 35000]})


res = pd.merge(df1, df2, how='left', on='고객ID')
print(res)