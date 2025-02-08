import pandas as pd

series = pd.Series([1, 2, None, 4, None, 6])
print(series.isnull())
series = series.fillna(0).astype('Int64')
print(series)