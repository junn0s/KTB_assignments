import pandas as pd

data = {
    '이름': ['김철수', '이영희', '김철수', '박민수'],
    '나이': [25, 30, 25, 40],
    '성별': ['남', '여', '남', '남']
}

df = pd.DataFrame(data)
df = df.drop_duplicates()
print(df)