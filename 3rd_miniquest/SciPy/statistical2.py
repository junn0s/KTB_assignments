import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
group_A = np.random.normal(loc=55, scale=8, size=200)  # 평균 55, 표준편차 8
group_B = np.random.normal(loc=60, scale=8, size=200)  # 평균 60, 표준편차 8

# DataFrame 생성: 두 그룹을 하나의 DataFrame에 결합하여 그룹 정보를 추가
df = pd.DataFrame({
    'value': np.concatenate([group_A, group_B]),
    'group': ['A'] * len(group_A) + ['B'] * len(group_B)
})

# Seaborn을 활용하여 히스토그램과 KDE 시각화
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='value', hue='group', bins=30, kde=True, stat="density", common_norm=False)
plt.title("Group A vs Group B: Distribution with KDE")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# 독립 표본 t-검정 수행: 두 그룹의 평균 차이가 유의미한지 검정
t_stat, p_value = stats.ttest_ind(group_A, group_B)

# 결과 출력
print("t-검정 통계량:", t_stat)
print("p-value:", p_value)
if p_value < 0.05:
    print("p-value가 0.05보다 작으므로, 두 그룹 간 평균 차이가 유의미합니다.")
else:
    print("p-value가 0.05 이상이므로, 두 그룹 간 평균 차이가 유의미하지 않습니다.")