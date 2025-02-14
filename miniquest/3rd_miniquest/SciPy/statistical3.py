import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 데이터 생성 (광고 A와 B의 클릭 여부)
observed_data = pd.DataFrame({
    "Ad_A": [120, 380],  # 광고 A: 클릭 120명, 미클릭 380명 (총 500명)
    "Ad_B": [150, 350]   # 광고 B: 클릭 150명, 미클릭 350명 (총 500명)
}, index=["Click", "No Click"])

# 전체 집계 계산
total_A = observed_data["Ad_A"].sum()           # 500
total_B = observed_data["Ad_B"].sum()           # 500
overall_total = total_A + total_B               # 1000
total_click = observed_data.loc["Click"].sum()  # 120 + 150 = 270
total_no_click = observed_data.loc["No Click"].sum()  # 380 + 350 = 730

# 기대값 계산 (귀무가설: 두 광고의 클릭률은 동일하다)
# 전체 클릭률 = 270 / 1000 = 0.27, 전체 미클릭률 = 730 / 1000 = 0.73
expected_A_click = total_A * (total_click / overall_total)      # 500 * 0.27 = 135
expected_A_no_click = total_A * (total_no_click / overall_total)  # 500 * 0.73 = 365
expected_B_click = total_B * (total_click / overall_total)        # 135
expected_B_no_click = total_B * (total_no_click / overall_total)    # 365

# 기대값과 관측값을 1차원 배열로 구성 (행 우선 순서)
expected_flat = np.array([expected_A_click, expected_A_no_click,
                          expected_B_click, expected_B_no_click])
observed_flat = np.array([observed_data.loc["Click", "Ad_A"],
                          observed_data.loc["No Click", "Ad_A"],
                          observed_data.loc["Click", "Ad_B"],
                          observed_data.loc["No Click", "Ad_B"]])

# stats.chisquare를 사용하여 카이제곱 검정 수행
chi2_stat, p_value = stats.chisquare(f_obs=observed_flat, f_exp=expected_flat)

if p_value < 0.05:
    print("p-value가 0.05보다 작으므로, 광고 A와 B의 클릭률 차이는 유의미합니다.")
else:
    print("p-value가 0.05 이상이므로, 광고 A와 B의 클릭률 차이는 유의미하지 않습니다.")

# 각 광고의 클릭률 계산
click_rate_A = observed_data.loc["Click", "Ad_A"] / total_A  # 120/500
click_rate_B = observed_data.loc["Click", "Ad_B"] / total_B  # 150/500

# 클릭률 데이터를 DataFrame으로 정리
click_rates_df = pd.DataFrame({
    "Ad": ["Ad A", "Ad B"],
    "Click_Rate": [click_rate_A, click_rate_B]
})

# Seaborn의 barplot을 사용하여 클릭률 비교 그래프 그리기
plt.figure(figsize=(8, 6))
# hue를 "Ad"로 지정하여 각 광고별 색상을 자동으로 부여하고, 범례는 제거
sns.barplot(x="Ad", y="Click_Rate", data=click_rates_df, hue="Ad", palette="viridis", dodge=False)
plt.legend([], [], frameon=False)  # 범례 제거
plt.ylim(0, 1)
plt.title("ad A vs ad B")
plt.ylabel("Click Rate")
plt.xlabel("ad")
plt.show()