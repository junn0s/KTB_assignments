import seaborn as sns
import matplotlib.pyplot as plt

# 예제 데이터 로드
tips = sns.load_dataset("tips")

# 성별에 따라 데이터 분리
tips_male = tips[tips['sex'] == 'Male']
tips_female = tips[tips['sex'] == 'Female']

# 남성 데이터에 대해 pairplot 생성 (요일에 따라 색상 적용)
g_male = sns.pairplot(data=tips_male,
                      vars=['total_bill', 'tip', 'size'],
                      hue='day',               # 요일(day)에 따라 색상 지정
                      palette='Set2',          # 요일별 색상 팔레트 설정
                      diag_kind='hist')        # 대각선에는 히스토그램 표시
g_male.fig.suptitle("Pairplot - Male", y=1.02)   # 전체 제목 설정

# 여성 데이터에 대해 pairplot 생성 (요일에 따라 색상 적용)
g_female = sns.pairplot(data=tips_female,
                        vars=['total_bill', 'tip', 'size'],
                        hue='day',             # 요일(day)에 따라 색상 지정
                        palette='Set2',        # 요일별 색상 팔레트 설정
                        diag_kind='hist')      # 대각선에는 히스토그램 표시
g_female.fig.suptitle("Pairplot - Female", y=1.02)  # 전체 제목 설정

# 생성된 두 개의 figure 표시
plt.show()