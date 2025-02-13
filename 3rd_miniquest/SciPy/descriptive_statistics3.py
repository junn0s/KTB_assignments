import numpy as np
import pandas as pd
import scipy.stats as stats

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)
df = pd.DataFrame(data, columns=["value"])

# 왜도, 첨도 계산
skewness = stats.skew(df["value"])
kurtosis = stats.kurtosis(df["value"])

print("skewness: ", skewness)
print("kurtosis: ", kurtosis)