import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import datetime

# 데이터 다운로드 (Stooq 사용)
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2023, 1, 1)
data = web.DataReader('AAPL', 'stooq', start, end)
data = data[::-1]  # Stooq는 내림차순으로 데이터를 제공하므로 순서를 뒤집음.

prices = data['Close'].values.reshape(-1, 1)

# 데이터 전처리
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

sequence_length = 10

class StockDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
        self.x, self.y = self.create_sequences()
        
    def create_sequences(self):
        x, y = [], []
        for i in range(len(self.data) - self.sequence_length):
            x.append(self.data[i:i+self.sequence_length])
            y.append(self.data[i+self.sequence_length])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# PyTorch 데이터셋 생성
dataset = StockDataset(scaled_prices, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 결과만 사용
        return out

# 모델 초기화
input_dim = 1  # 가격 데이터는 단일 차원
hidden_dim = 50  # 은닉 상태 차원
num_layers = 2  # LSTM 레이어 수
output_dim = 1  # 예측할 가격 (단일 값)

model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()  # MSE 손실 함수 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)  # 모델에 입력
        loss = criterion(outputs, targets)  # 손실 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.6f}')

evaluate(model, dataloader)



def predict_future(model, data, days=10):
    model.eval()
    data = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(days):
            pred = model(data).item()
            predictions.append(pred)
            
            # 시퀀스를 업데이트 (Sliding Window)
            new_seq = torch.cat([data[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32).to(device)], dim=1)
            data = new_seq

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))  # 원래 가격 범위로 복구

predicted_prices = predict_future(model, scaled_prices, days=10)

print("Predicted Prices:")
for i, price in enumerate(predicted_prices.flatten(), start=1):
    print(f"Day {i}: {price:.2f}")