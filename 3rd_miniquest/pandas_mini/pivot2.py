import pandas as pd

data = {
    '카테고리': ['전자', '가전', '전자', '가전'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}

df = pd.DataFrame(data)
res = pd.pivot_table(df, index='카테고리', columns='제품', values='판매량', aggfunc='sum')
print(res)