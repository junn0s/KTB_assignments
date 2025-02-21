# 가상 데이터셋을 생성한 뒤, 데이터셋을 학습, 검증, 테스트 데이터셋으로 분할

import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터
data = {
    'feature1': range(1, 1001),
    'feature2': range(1001, 2001),
    'label': [1 if x % 2 == 0 else 0 for x in range(1, 1001)]
}
df = pd.DataFrame(data)

# 데이터 분할 (학습, 검증, 테스트 - 6:2:2)
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("train data :", len(train_data))
print("validation data :", len(validation_data))
print("test data :", len(test_data))