# 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 샘플 데이터 생성
data = {
    'feature1': range(1, 101),
    'feature2': range(101, 201),
    'label': [1 if x % 2 == 0 else 0 for x in range(1, 101)]
}
df = pd.DataFrame(data)

# 데이터셋을 학습, 검증, 테스트 세트로 분할 (60:20:20 비율)
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


# 학습 데이터와 레이블 분리
X_train = train_data[['feature1', 'feature2']]
y_train = train_data['label']
X_validation = validation_data[['feature1', 'feature2']]
y_validation = validation_data['label']
X_test = test_data[['feature1', 'feature2']]
y_test = test_data['label']

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 검증 데이터로 성능 평가
y_val_pred = model.predict(X_validation)
val_accuracy = accuracy_score(y_validation, y_val_pred)
print("검증 데이터 정확도:", val_accuracy)

# 테스트 데이터로 최종 성능 평가
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("테스트 데이터 정확도:", test_accuracy)