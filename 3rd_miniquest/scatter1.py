import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [3, 1, 4, 5, 2]

plt.scatter(x, y, marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.title('scatter graph')
plt.show()