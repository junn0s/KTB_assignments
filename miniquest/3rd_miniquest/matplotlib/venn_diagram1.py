import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# 두 개의 과일 집합 정의
set_A = {"사과", "바나나", "체리", "망고"}
set_B = {"바나나", "망고", "포도", "수박"}

symmetric_diff = set_A ^ set_B
print(symmetric_diff)
