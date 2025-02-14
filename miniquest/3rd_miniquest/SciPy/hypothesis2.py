import numpy as np
import scipy.stats as stats

# 관측된 데이터 (Observed)
observed = np.array([50, 60, 90])

# 기대값 (Expected)
expected = np.array([66, 66, 66]) * (observed.sum() / np.sum([66, 66, 66]))

# chi square test
chi_state, p_value = stats.chisquare(observed, expected)

if p_value < 0.05:
    print("유의미한 차이가 있음")
else:
    print("유의미한 차이가 없음")