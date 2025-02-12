import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C', 'D', 'E']
values_2023 = [10, 15, 20, 25, 30]
values_2024 = [5, 10, 12, 18, 22]

plt.bar(categories, values_2023, color='dodgerblue', label='2023')
plt.bar(categories, values_2024, color='orange', bottom=values_2023, label='2024')
plt.xlabel("Category") 
plt.ylabel("Value")  
plt.title("Stacked Bar Chart")  
plt.legend()  
plt.show()