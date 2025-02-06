import pandas as pd

data = {'a': 100, 'b': 200, 'c': 300}
series = pd.Series(data)
print(series['b'])