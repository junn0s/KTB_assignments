import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

x = 80
cdf_value = stats.norm.cdf(x, loc=70, scale=8)
ppf_value = stats.norm.ppf(0.95, loc=70, scale=8)
print(cdf_value)
print(ppf_value)