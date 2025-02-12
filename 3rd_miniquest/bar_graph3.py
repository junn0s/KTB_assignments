import matplotlib.pyplot as plt
import numpy as np

departments = ['Sales', 'Marketing', 'IT', 'HR', 'Finance']
performance_2023 = [80, 70, 90, 60, 75]
performance_2024 = [85, 75, 95, 65, 80]

bar_width = 0.4
x = np.arange(len(departments))
plt.bar(x - bar_width/2, performance_2023, width=bar_width, color='skyblue', label='2023')
plt.bar(x + bar_width/2, performance_2024, width=bar_width, color='lightpink', label='2024')
plt.xlabel("departments")
plt.ylabel("performance")
plt.title("group bar chart")
plt.legend()
plt.show()