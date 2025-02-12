import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
group_x = np.random.randn(50) * 2  # 표준편차 2, 평균 0
group_y = np.random.randn(50) * 2 + 5  # 표준편차 2, 평균 5

plt.boxplot([group_x, group_y],
            patch_artist=True,  
            boxprops=dict(facecolor="lightblue", color="blue"),
            whiskerprops=dict(color="red", linewidth=2),  
            capprops=dict(color="green", linewidth=2), 
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker='o', color='red', markersize=8))  

plt.title("Box Plot with Outliers Highlighted") 
plt.xticks([1, 2], ["group_x", "group_y"])
plt.ylabel("Values")  
plt.show() 