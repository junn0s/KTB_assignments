import numpy as np
import scipy.stats as stats

# 샘플 데이터 생성
np.random.seed(42)
group_1 = np.random.normal(loc=50, scale=10, size=30)  
group_2 = np.random.normal(loc=55, scale=10, size=30)  
group_3 = np.random.normal(loc=60, scale=10, size=30) 

# ANOVA 분석
f_statistic, p_value = stats.f_oneway(group_1, group_2, group_3)

if p_value < 0.05:
    print("그룹 간 평균에 유의미한 차이가 있다")
else:
    print("그룹 간 평균에 유의미한 차이가 없다")