import numpy as np
import pandas as pd


np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)  
df = pd.DataFrame(data, columns=["value"])

mean = df["value"].mean()
median = df["value"].median()
print(mean - median)