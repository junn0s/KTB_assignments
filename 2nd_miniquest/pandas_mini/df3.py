import pandas as pd

data = {'이름': ['홍길동', '김철수', '박영희', '이순신'], 
        '국어': [85, 90, 88, 92], 
        '영어': [78, 85, 89, 87], 
        '수학': [92, 88, 84, 90]}

df = pd.DataFrame(data)
df['총점'] = df['국어'] + df['영어'] + df['수학']
print(df[df['총점'] >= 250])