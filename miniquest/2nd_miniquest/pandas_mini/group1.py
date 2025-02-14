import pandas as pd

data = {
    '이름': ['홍길동', '김철수', '박영희', '이순신'],
    '부서': ['영업', '영업', '인사', '인사'],
    '급여': [5000, 5500, 4800, 5100]
}

df = pd.DataFrame(data)
print(df.groupby('부서')['급여'].sum())