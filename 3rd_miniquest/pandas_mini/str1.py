import pandas as pd
data = pd.Series(["HELLO", "WORLD", "PYTHON", "PANDAS"])

data = data.str.lower()
print(data)