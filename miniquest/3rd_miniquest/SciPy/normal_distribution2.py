import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

x = 65
pdf_value = stats.norm.pdf(x, loc=50, scale=10)
print(pdf_value)