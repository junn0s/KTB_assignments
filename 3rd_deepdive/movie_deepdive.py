import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.0f' % x)  # 과학적 표기법(e등) 말고 정수 형태로 출력하는 코드

# --------------------------------------------------------
# 1) 영화 데이터
# --------------------------------------------------------
# 날짜 범위 (2025-02-14 ~ 2025-02-20, 7일)
dates = pd.date_range('2025-02-14', periods=7)

# 영화 3종, 상영관 3종
movie_names = ['Memento', 'Inception', 'Tenet']
theaters = ['CGV', 'Lotte cinema', 'Megabox']

# 랜덤 데이터 생성
records = []
for d in dates:
    for m in movie_names:
        for t in theaters:
            ticket_sold = np.random.randint(500, 2000)  # 500 ~ 2000장 사이 랜덤
            revenue = ticket_sold * 10000                # 티켓 1장당 10000원
            records.append([d, m, t, ticket_sold, revenue])

df = pd.DataFrame(records, columns=['date', 'movie_names', 'theater', 'ticket_sold', 'revenue'])


# --------------------------------------------------------
# 2) 시계열 데이터 변환
# --------------------------------------------------------
# date를 시계열로 변환
df['date'] = pd.to_datetime(df['date'])

print(">> 원본 데이터프레임 확인 <<")
print(df, "\n")


# --------------------------------------------------------
# 3) GroupBy를 이용한 집계
#    - theater, movie_names별 수익 합계
# --------------------------------------------------------
grouped = df.groupby(['theater', 'movie_names'])['revenue'].sum().reset_index()
print(">> 상영관 & 영화별 총 매출 <<")
print(grouped, "\n")


# --------------------------------------------------------
# 4) 날짜를 인덱스로 지정 후 일 단위 resample
# --------------------------------------------------------
df_indexed = df.set_index('date')
df_daily = df_indexed.resample('D')['revenue'].sum()  # 일 단위 집계
print(">> 일 단위 매출 합계 <<")
print(df_daily, "\n")


# --------------------------------------------------------
# 5) Pivot & Pivot Table
# --------------------------------------------------------
# 날짜(date)와 영화(movie_names)를 함께 index로, 상영관(theater)을 columns로 사용
df_reset = df.reset_index(drop=True)  # 기존 index를 버리고 새 index 부여

df_pivot = df_reset.pivot(
    index=['date', 'movie_names'],
    columns='theater',
    values='revenue'
)
print(">> pivot 결과 <<")
print(df_pivot.head(10), "\n")


# pivot_table(): 중복이 있어도 aggfunc(aggregation function)로 집계 처리 가능
df_pivot_table = pd.pivot_table(
    df_reset,
    index='movie_names',
    columns='theater',
    values='revenue',
    aggfunc=['sum', 'mean', 'max', 'min']
)
print(">> pivot_table 결과 <<")
print(df_pivot_table, "\n")


# --------------------------------------------------------
# 6) Melt: Wide → Long 형태 변환
#    - pivot 결과를 다시 long 형으로
# --------------------------------------------------------
df_melt = pd.melt(
    df_pivot.reset_index(),
    id_vars=['date', 'movie_names'],
    var_name='theater',
    value_name='revenue'
)
print(">> melt 결과 (wide → long) <<")
print(df_melt, "\n")


# --------------------------------------------------------
# 7) Stack/Unstack 예시 (멀티인덱스)
# --------------------------------------------------------
# date, movie_names, theater를 인덱스로 묶어둔 상태에서 ticket_sold와 revenue만 남김
df_multi = df_reset.set_index(['date', 'movie_names', 'theater'])[['ticket_sold', 'revenue']]
print(">> 멀티 인덱스 DataFrame <<")
print(df_multi.head(9), "\n")

# 7-1) stack() → 열을 행의 하위 레벨로 쌓기
df_stacked = df_multi.stack()
print(">> df_multi.stack() 결과 <<")
print(df_stacked.head(12), "\n")

# 7-2) unstack() → 행의 하위 레벨을 열로 펼치기 (원상복구)
df_unstacked = df_stacked.unstack()
print(">> df_stacked.unstack() 결과 (원상복구) <<")
print(df_unstacked.head(9), "\n")