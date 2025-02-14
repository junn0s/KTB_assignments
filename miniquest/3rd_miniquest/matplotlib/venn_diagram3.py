import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# 두 개의 과일 집합 정의
set_A = {"사과", "바나나", "체리", "망고"}
set_B = {"바나나", "망고", "포도", "수박"}

# 벤 다이어그램 생성
venn_diagram = venn2([set_A, set_B], set_labels=('Set A', 'Set B'))

# 교집합 (두 집합 모두에 존재하는 요소) 계산
intersection = set_A & set_B

# 교집합 개수가 2개 이상이면 교집합 영역 색상 변경
if len(intersection) >= 2:
    intersection_patch = venn_diagram.get_patch_by_id('11')
    if intersection_patch is not None:
        intersection_patch.set_color('yellow')
        intersection_patch.set_alpha(0.5)

plt.title("Venn Diagram with Conditional Coloring")
plt.show()