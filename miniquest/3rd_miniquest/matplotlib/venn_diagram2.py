# 세 개의 과일 집합 정의
set_A = {"사과", "바나나", "체리", "망고"}
set_B = {"바나나", "망고", "포도", "수박"}
set_C = {"망고", "수박", "딸기", "오렌지"}

# 각 집합에만 존재하는 요소 (단독 요소) 계산
only_A = set_A - (set_B | set_C)
only_B = set_B - (set_A | set_C)
only_C = set_C - (set_A | set_B)

print("Set A 단독 요소:", only_A, "개수:", len(only_A))
print("Set B 단독 요소:", only_B, "개수:", len(only_B))
print("Set C 단독 요소:", only_C, "개수:", len(only_C))

# 쌍 집합 간의 교집합 계산
intersection_AB = set_A & set_B
intersection_AC = set_A & set_C
intersection_BC = set_B & set_C

print("Set A ∩ Set B:", intersection_AB, "개수:", len(intersection_AB))
print("Set A ∩ Set C:", intersection_AC, "개수:", len(intersection_AC))
print("Set B ∩ Set C:", intersection_BC, "개수:", len(intersection_BC))

# 세 집합 모두의 교집합 계산
intersection_ABC = set_A & set_B & set_C
print("Set A ∩ Set B ∩ Set C:", intersection_ABC, "개수:", len(intersection_ABC))