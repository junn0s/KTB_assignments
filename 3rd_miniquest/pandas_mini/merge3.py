import pandas as pd

df1 = pd.DataFrame({
    '고객ID': [1, 2, 3],
    '도시': ['서울', '부산', '대전'],
    '구매액': [10000, 20000, 30000]
})

df2 = pd.DataFrame({
    '고객ID': [1, 2, 3],
    '도시': ['서울', '부산', '광주'],
    '구매액': [15000, 25000, 35000]
})


res = pd.merge(df1, df2, on=['고객ID', '도시'], suffixes=('_기존', '_신규'))
print(res)