import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
filtered_tips = tips[tips['sex'] == 'Female']

sns.regplot(x='total_bill',
            y='tip',
            data=filtered_tips,
            scatter_kws={'alpha': 0.5})

plt.title("Relationship between Total Bill and Tip (Female only)")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show()