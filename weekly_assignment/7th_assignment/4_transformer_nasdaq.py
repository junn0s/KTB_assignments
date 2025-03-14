# 나스닥 대표주 3가지 (애플, 엔비, 테슬라)
# 2024년까지 학습 후 25년 1월부터 3월 12일까지 예측 모델

import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

# 애플, 엔비디아, 테슬라 주가 데이터 (약 5년간)
stocks = ["AAPL", "NVDA", "TSLA"]
data = {}
for stock in stocks:
    stock_data = yf.download(stock, start="2020-01-01", end="2025-03-12", auto_adjust=True)
    print(stock_data.columns)
    stock_data['MA7'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['MA15'] = stock_data['Close'].rolling(window=15).mean()
    stock_data['MA30'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['MA60'] = stock_data['Close'].rolling(window=60).mean()
    stock_data['MA90'] = stock_data['Close'].rolling(window=90).mean()
    stock_data.dropna(inplace=True)
    data[stock] = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA15', 'MA30', 'MA60', 'MA90']]

# 평탄화, 스케일링
df = pd.concat(data, axis=1)
def flatten_columns(columns):
    new_cols = []
    for col in columns:
        if isinstance(col, tuple):
            if len(col) == 2:
                price, ticker = col
                new_cols.append(f"{ticker}_{price}")
            else:
                new_cols.append("_".join(map(str, col)))
        else:
            new_cols.append(col)
    return new_cols

df.columns = flatten_columns(df.columns)
df.fillna(method='ffill', inplace=True)

train_df = df.loc[:'2024-12-31']    
test_df  = df.loc['2025-01-01':'2025-03-12']    

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

num_stocks = len(stocks)
features_per_stock = 10
total_features = num_stocks * features_per_stock

# 시퀀스 생성
def create_sequences(data, seq_length=30, num_stocks=3, features_per_stock=10):
    sequences, targets = [], []
    close_indices = [i * features_per_stock + 3 for i in range(num_stocks)]
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, close_indices])  
    return np.array(sequences), np.array(targets)

X_train, y_train = create_sequences(train_scaled, seq_length=30, num_stocks=len(stocks), features_per_stock=10)
combined_test = np.concatenate([train_scaled[-30:], test_scaled], axis=0) 
X_test, y_test = create_sequences(combined_test, seq_length=30, num_stocks=len(stocks), features_per_stock=10)

test_dates = test_df.index 

# Dataset 정의
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 포지셔널 인코딩
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 트랜스포머
class TransformerStockPredictor(nn.Module):
    def __init__(self, input_dim=total_features, num_heads=5, num_layers=2, hidden_dim=128, num_stocks=len(stocks)):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_stocks)
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# 파라미터
num_stocks = len(stocks)
features_per_stock = 10
total_features = num_stocks * features_per_stock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerStockPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 학습
def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

train_model(model, train_loader, criterion, optimizer, device)

# 평가 및 예측
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            predictions.append(outputs)
            actuals.append(y_batch.numpy())
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    return predictions, actuals

predictions, actuals = evaluate_model(model, test_loader, device)

# 시각화: 테스트 데이터 기간 (2025-01-01 ~ 2025-03-12)
plt.figure(figsize=(12, 6))
for i, stock in enumerate(stocks):
    plt.plot(test_dates, actuals[:, i], label=f"Actual {stock}", linestyle='dashed')
    plt.plot(test_dates, predictions[:, i], label=f"Predicted {stock}")
plt.legend()
plt.title("Stock Price Prediction (2025-01-01 ~ 2025-03-12)")
plt.show()