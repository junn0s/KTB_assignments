import pandas as pd
from sklearn.model_selection import train_test_split

# 샘플 데이터 생성
data = {
    'feature1': range(1, 101),
    'feature2': range(101, 201),
    'label': [1 if x % 2 == 0 else 0 for x in range(1, 101)]
}
df = pd.DataFrame(data)


train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("train data :", len(train_data))
print("validation data :", len(validation_data))
print("test data :", len(test_data))
