import pandas as pd

data = {
    '제품': ['노트북', '태블릿', '노트북', '스마트폰'],
    '가격': [1500000, 800000, 1500000, 1000000],
    '카테고리': ['전자기기', '전자기기', '전자기기', '전자기기']
}

df = pd.DataFrame(data)
df = df.drop_duplicates(subset=['제품'])
print(df)