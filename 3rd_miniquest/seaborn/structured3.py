import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "category": ["A", "A", "B", "B", "C", "C", "C", "A", "B", "C"],
    "score": [80, 85, 70, 75, 95, 90, 100, 82, 72, 98]
})

sns.violinplot(x='category', y='score', data=data)
sns.stripplot(x='category', y='score', data=data)
plt.title('score by category')
plt.show()