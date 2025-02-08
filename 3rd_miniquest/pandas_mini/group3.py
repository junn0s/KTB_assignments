import pandas as pd

data = {
    '이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬', '신사임당'],
    '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
    '급여': [5000, 5500, 4800, 5100, 6000, 6200]
}

df = pd.DataFrame(data)
print(df.groupby('부서').filter(lambda x:x['급여'].mean() >= 5000))