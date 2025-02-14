import numpy as np
import scipy.stats as stats

# 샘플 데이터 생성
np.random.seed(42)
sample_data = np.random.normal(loc=50, scale=5, size=30)

# t-test 수행
x = 52
t_state, p_value = stats.ttest_1samp(sample_data, x)

# 유의수준 설정
alpha = 0.05

if p_value < 0.05:
    print("평균 52와 유의미한 차이가 있음")
else:
    print("평균 52와 유의미한 차이가 없음")