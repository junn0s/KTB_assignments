# OR, NOT 연산 모델링

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(precision=4, sci_mode=False)


# OR 데이터 정의
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [1]], dtype=np.float32)

# NOT 데이터 정의
X2 = np.array([[0], [1]], dtype=np.float32)
y2 = np.array([[1], [0]], dtype=np.float32)

# Tensor로 변환
X = torch.tensor(X)
y = torch.tensor(y)
X2 = torch.tensor(X2)
y2 = torch.tensor(y2)


class ORModel(nn.Module):
    def __init__(self):
        super(ORModel, self).__init__()  # nn.model(부모) 초기화
        self.layer1 = nn.Linear(2, 4)  # 첫 번째 선형 레이어, 입력 크기 2, 출력 크기 4
        self.layer2 = nn.Linear(4, 1)  # 두 번째 선형 레이어, 입력 크기 4, 출력 크기 1
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 첫 번째 레이어와 ReLU 적용
        x = self.sigmoid(self.layer2(x))  # # 두 번째 레이어와 Sigmoid 적용
        return x

class NOTModel(nn.Module):
    def __init__(self):
        super(NOTModel, self).__init__()
        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 1)  
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 첫 번째 레이어와 ReLU 적용
        x = self.sigmoid(self.layer2(x))  # # 두 번째 레이어와 Sigmoid 적용
        return x

model = ORModel()
model2 = NOTModel()


criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

num_epochs = 1000  # 총 에포크 수

for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    model2.train()
    optimizer.zero_grad()  # 옵티마이저의 변화도(gradient)를 초기화
    optimizer2.zero_grad()
    outputs = model(X)  # 모델에 입력 데이터를 넣어 출력 계산
    outputs2 = model2(X2)
    loss = criterion(outputs, y)  # 출력과 실제 레이블을 비교하여 손실 계산
    loss2 = criterion(outputs2, y2)
    loss.backward()  # 역전파를 통해 손실에 대한 그래디언트 계산
    loss2.backward()
    optimizer.step()  # 옵티마이저가 매개변수를 업데이트
    optimizer2.step()



model.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():  # 평가 중에는 그래디언트를 계산하지 않음
    outputs = model(X)  # 모델에 입력 데이터를 전달하여 출력값 계산
    predicted = (outputs > 0.5).float()  # 출력값이 0.5보다 크면 1, 아니면 0으로 변환 (이진 분류)
    accuracy = (predicted == y).float().mean()  # 예측값과 실제값을 비교하여 정확도 계산
    loss = criterion(outputs, y)  # 손실 함수(크로스 엔트로피 손실)를 사용하여 손실 계산
    print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')  # 손실과 정확도 출력

model2.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():  # 평가 중에는 그래디언트를 계산하지 않음
    outputs2 = model2(X2)  # 모델에 입력 데이터를 전달하여 출력값 계산
    predicted2 = (outputs2 > 0.5).float()  # 출력값이 0.5보다 크면 1, 아니면 0으로 변환 (이진 분류)
    accuracy2 = (predicted2 == y2).float().mean()  # 예측값과 실제값을 비교하여 정확도 계산
    loss2 = criterion(outputs2, y2)  # 손실 함수(크로스 엔트로피 손실)를 사용하여 손실 계산
    print(f'Loss: {loss2.item()}, Accuracy: {accuracy2.item()}')  # 손실과 정확도 출력
    
    
with torch.no_grad():
    predictions = model(X)
    print(f'Predictions: {predictions}')

with torch.no_grad():
    predictions = model2(X2)
    print(f'Predictions: {predictions}')