import pandas as pd

data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}

df = pd.DataFrame(data)
res = df.pivot(index=['날짜'], columns=['제품'], values=['판매량'])
print(res)