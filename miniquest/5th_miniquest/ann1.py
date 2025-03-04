import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.input = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  
    
    def forward(self, x):
        x = self.relu(self.input(x))  
        x = self.sigmoid(self.output(x))  
        return x

model = XORModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(X)
    rounded_predictions = torch.round(predictions)
    for i in range(len(X)):
        print(f"입력: {X[i].tolist()} => 예측: {rounded_predictions[i].item()}")