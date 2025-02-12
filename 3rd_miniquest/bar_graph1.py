import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D', 'E']
values = [12, 25, 18, 30, 22]

plt.bar(categories, values, color='skyblue')
plt.xlabel('categories')
plt.ylabel('values')
plt.title('basic bar graph')
plt.show()