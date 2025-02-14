import pandas as pd

data = {
    'Name': ['Alice', 'Bob', None, 'Charlie'],
    'Age': [25, None, 28, 35],
    'City': ['New York', None, 'Chicago', None]
}

df = pd.DataFrame(data)
# 이름 채우기
df['Name'] = df['Name'].fillna("Unknown")
# 나이 채우기
avg_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(avg_age)

print(df)