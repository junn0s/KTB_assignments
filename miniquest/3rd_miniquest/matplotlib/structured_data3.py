import pandas as pd

data = {
    '이름': ['김철수', '이영희', '박민수', '최지현', '홍길동'],
    '나이': [25, 30, 35, 28, 40],
    '직업': ['개발자', '마케터', '개발자', '디자이너', 'CEO'],
    '연봉': [4000, 3500, 5000, 4200, 10000],
    '가입일': ['2020-05-21', '2019-07-15', '2021-01-10', '2018-11-03', '2017-09-27']
}

df = pd.DataFrame(data)
df['가입일'] = pd.to_datetime(df['가입일'])
condition = df['가입일'].dt.year <= 2019
df.loc[condition, '연봉'] = (df.loc[condition, '연봉'] * 1.1).astype(int)
average_salary = df['연봉'].mean()
print(average_salary)