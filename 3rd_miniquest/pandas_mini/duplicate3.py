import pandas as pd

data = {
    '학생': ['김민수', '박지현', '김민수', '이정훈'],
    '성적': [90, 85, 90, 88],
    '학교': ['A고', 'B고', 'A고', 'C고']
}

df = pd.DataFrame(data)
df_unique = df.drop_duplicates()
df_unique.to_csv('unique_data.csv', index=False)
df_loaded = pd.read_csv('unique_data.csv')
print(df_loaded)