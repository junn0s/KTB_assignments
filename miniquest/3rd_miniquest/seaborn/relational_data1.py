import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(7, 5))
sns.scatterplot(x="total_bill", y="tip", data=tips, color="skyblue")
plt.title("Scatter Plot: Total Bill vs. Tip")  
plt.xlabel("Total Bill ($)")  
plt.ylabel("Tip ($)")
plt.show()