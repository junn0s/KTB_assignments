# 어떤 논리연산이라도 모델로 처리하는 AI 서버 제작

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(precision=4, sci_mode=False)

# Gate type one-hot encoding
# 순서: [AND, OR, XOR, NOT]
gate_types = {
    'AND': [1, 0, 0, 0],
    'OR':  [0, 1, 0, 0],
    'XOR': [0, 0, 1, 0],
    'NOT': [0, 0, 0, 1]
}

# 데이터: 각 게이트에 대해 (입력, 출력) 쌍 생성
def create_data():
    data = []
    
    # AND
    and_truth = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 1
    }
    for (x, y), out in and_truth.items():
        inp = [x, y] + gate_types['AND']
        data.append((inp, out))
        
    # OR
    or_truth = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1
    }
    for (x, y), out in or_truth.items():
        inp = [x, y] + gate_types['OR']
        data.append((inp, out))
    
    # XOR
    xor_truth = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }
    for (x, y), out in xor_truth.items():
        inp = [x, y] + gate_types['XOR']
        data.append((inp, out))
        
    # NOT
    not_truth = {
        (0,): 1,
        (1,): 0
    }
    for (x,), out in not_truth.items():
        inp = [x, 0] + gate_types['NOT']
        data.append((inp, out))
        
    return data

# 데이터셋 생성
dataset = create_data()
inputs = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
labels = torch.tensor([[d[1]] for d in dataset], dtype=torch.float32)

# 통합 모델
class LogicGateModel(nn.Module):
    def __init__(self):
        super(LogicGateModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

model = LogicGateModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        with torch.no_grad():
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == labels).float().mean()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# 평가
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == labels).float().mean()
    print("\nFinal Evaluation:")
    print("Detailed Predictions:")
    for inp, label, raw, pred in zip(inputs, labels, outputs, predicted):
        x1, x2 = inp[0].item(), inp[1].item()
        gate_onehot = inp[2:6]
        
        if torch.allclose(gate_onehot, torch.tensor(gate_types['AND'], dtype=torch.float32)):
            gate = "AND"
        elif torch.allclose(gate_onehot, torch.tensor(gate_types['OR'], dtype=torch.float32)):
            gate = "OR"
        elif torch.allclose(gate_onehot, torch.tensor(gate_types['XOR'], dtype=torch.float32)):
            gate = "XOR"
        elif torch.allclose(gate_onehot, torch.tensor(gate_types['NOT'], dtype=torch.float32)):
            gate = "NOT"
        else:
            gate = "Unknown"
        
        print(f"Gate: {gate:4s} | Input: [{x1:.0f}, {x2:.0f}] | Label: {label.item():.0f} | Binary: {pred.item():.0f}")
    print(f"Accuracy: {accuracy.item():.4f}")


