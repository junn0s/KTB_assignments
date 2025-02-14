import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

date_rng = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
df = pd.DataFrame({
    "datetime": date_rng,
    "value": np.random.randint(50, 150, size=len(date_rng))
})

df.set_index("datetime", inplace=True)
df["EMA_7"] = df["value"].ewm(span=7, adjust=False).mean()

print(df.head(10).reset_index())