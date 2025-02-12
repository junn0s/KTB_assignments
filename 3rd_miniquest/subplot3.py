import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 데이터 생성
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.random.randn(100)
categories = ['A', 'B', 'C', 'D', 'E']
values = [3, 7, 5, 2, 8]

# Figure와 그리드스펙 생성 (3행 3열 그리드)
fig = plt.figure(constrained_layout=True, figsize=(10, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

# 1. 선 그래프: Figure의 첫 번째 행 전체 중 왼쪽 2열 (행:0, 열:0~1)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x, y1, color='skyblue', marker='o')
ax1.set_title("Line Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("sin(x)")

# 2. 산점도: 첫 번째 행의 오른쪽 1열 (행:0, 열:2)
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(x, y2, color='orange')
ax2.set_title("Scatter Plot")
ax2.set_xlabel("x")
ax2.set_ylabel("Random Data")

# 3. 막대 그래프: 왼쪽 아래 영역, 두 번째 행부터 마지막 행까지, 첫 번째 열 (행:1~2, 열:0)
ax3 = fig.add_subplot(gs[1:, 0])
ax3.bar(categories, values, color='lightpink')
ax3.set_title("Bar Graph")
ax3.set_xlabel("Category")
ax3.set_ylabel("Value")

# 4. 히스토그램: 오른쪽 아래 영역, 두 번째 행부터 마지막 행까지, 나머지 열 (행:1~2, 열:1~2)
ax4 = fig.add_subplot(gs[1:, 1:])
ax4.hist(y2, bins=20, color='purple', edgecolor='yellow')
ax4.set_title("Histogram")
ax4.set_xlabel("Value")
ax4.set_ylabel("Frequency")

plt.show()