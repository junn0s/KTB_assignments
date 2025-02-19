# XOR 문제 퍼셉트론으로 해결
# 선형 분리가 불가능하므로, 단일 퍼셉트론으로는 해결할 수 없음 (은닉층 한개 필요)

import numpy as np

# XOR 데이터 정의
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 신경망 구조 설정: 2-2-1
input_size = 2
hidden_size = 2
output_size = 1

# 가중치와 편향 초기화 (랜덤 초기화)
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# 활성화 함수: Sigmoid 및 그 도함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

learning_rate = 0.1
epochs = 10000

# 학습 시작 (역전파를 통한 업데이트)
for epoch in range(epochs):
    total_loss = 0
    # 각 데이터 포인트에 대해 학습 (온라인 방식)
    for i in range(len(X)):
        # --- 순전파 (Forward Pass) ---
        x_i = X[i].reshape(1, input_size)  # shape (1,2)
        y_i = y[i].reshape(1, output_size)   # shape (1,1)

        # 은닉층 계산
        z1 = np.dot(x_i, W1) + b1            # shape (1, hidden_size)
        a1 = sigmoid(z1)                     # shape (1, hidden_size)

        # 출력층 계산
        z2 = np.dot(a1, W2) + b2             # shape (1, output_size)
        a2 = sigmoid(z2)                     # shape (1, output_size)

        # 손실 계산 (이진 분류에서 cross-entropy loss를 사용할 수 있지만,
        # 여기서는 계산과 미분이 간단한 평균 제곱 오차(MSE)를 사용
        loss = 0.5 * (y_i - a2) ** 2
        total_loss += loss

        # --- 역전파 (Backward Pass) ---
        # 출력층에서의 오차 (MSE의 미분: (a2 - y))
        delta2 = a2 - y_i                    # shape (1, output_size)
        # 은닉층-출력층 가중치, 편향에 대한 기울기
        dW2 = np.dot(a1.T, delta2)           # shape (hidden_size, output_size)
        db2 = delta2.flatten()               # shape (output_size,)

        # 은닉층 오차: 출력 오차를 W2를 통해 역전파하고, Sigmoid의 도함수 적용
        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(z1)  # shape (1, hidden_size)
        dW1 = np.dot(x_i.T, delta1)          # shape (input_size, hidden_size)
        db1 = delta1.flatten()               # shape (hidden_size,)

        # --- 가중치 및 편향 업데이트 ---
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # 1000 에포크마다 평균 손실 출력
    if (epoch + 1) % 1000 == 0:
        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss.item():.4f}")

print("\n학습 완료!")
print("학습된 W1:", W1)
print("학습된 b1:", b1)
print("학습된 W2:", W2)
print("학습된 b2:", b2)

# 예측 함수 정의
def predict(x):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2 > 0.5

# 학습된 모델로 XOR 문제 예측
for input_data in X:
    pred = predict(input_data.reshape(1, input_size))
    print(f"입력: {input_data}, 예측 출력: {int(pred)}")