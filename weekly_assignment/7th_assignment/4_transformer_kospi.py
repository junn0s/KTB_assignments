# 코스피 데이터로 지수 예측
# 2024년까지 학습 후 25년 1월부터 3월 12일까지 예측 모델

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pykrx import stock

df_trainval = stock.get_index_ohlcv_by_date("20100101", "20241231", "1001")
df_test = stock.get_index_ohlcv_by_date("20250101", "20250312", "1001")

train_df = df_trainval[["시가", "고가", "저가", "종가", "거래량", "거래대금", "상장시가총액"]]
test_df = df_test[["시가", "고가", "저가", "종가", "거래량", "거래대금", "상장시가총액"]]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

price_scaler = MinMaxScaler()
price_scaler.fit(train_df[['종가']])

def create_sequences(data, seq_length=30):
    sequences, targets = [], []
    close_idx = 3  # 종가 인덱스
    for i in range(len(data) - seq_length):
        seq_x = data[i:i+seq_length]
        seq_y = data[i+seq_length, close_idx]
        sequences.append(seq_x)
        targets.append(seq_y)
    return np.array(sequences), np.array(targets)

X_train, y_train = create_sequences(train_scaled, seq_length=30)
combined_test = np.concatenate([train_scaled[-30:], test_scaled], axis=0)
X_test, y_test = create_sequences(combined_test, seq_length=30)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TransformerStockPredictor(nn.Module):
    def __init__(self, input_dim=7, d_model=32, num_heads=2, num_layers=2, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerStockPredictor(input_dim=7, d_model=32, num_heads=2, num_layers=2, hidden_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

train_model(model, train_loader, criterion, optimizer, device, epochs=150)

def evaluate_model(model, loader, device):
    model.eval()
    preds, reals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
            reals.append(y_batch.numpy())
    preds = np.concatenate(preds, axis=0)
    reals = np.concatenate(reals, axis=0)
    return preds, reals

preds, reals = evaluate_model(model, test_loader, device)

test_dates = test_df.index[-len(preds):]
preds_inversed = price_scaler.inverse_transform(preds.reshape(-1, 1))
reals_inversed = price_scaler.inverse_transform(reals.reshape(-1, 1))

plt.figure(figsize=(12,5))
plt.plot(test_dates, reals_inversed.flatten(), label="Real KOSPI", linestyle='--')
plt.plot(test_dates, preds_inversed.flatten(), label="Predicted KOSPI")
plt.title("KOSPI Prediction (2025-01-01 ~ 2025-03-12)")
plt.legend()
plt.show()