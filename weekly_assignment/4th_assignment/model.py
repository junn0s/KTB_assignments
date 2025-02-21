import torch
import torch.nn as nn
import torch.optim as optim

class LogicGateAI:
    def __init__(self, num_epochs=1000, lr=0.01):
        # Gate type one-hot encoding (순서: [AND, OR, XOR, NOT])
        self.gate_types = {
            'AND': [1, 0, 0, 0],
            'OR':  [0, 1, 0, 0],
            'XOR': [0, 0, 1, 0],
            'NOT': [0, 0, 0, 1]
        }
        self.num_epochs = num_epochs
        self.lr = lr
        
        # 데이터셋 생성
        self.inputs, self.labels = self.create_data()
        
        # 모델 구축 및 학습 관련 요소 초기화
        self.model = self.build_model()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def create_data(self):
        """각 논리 게이트(AND, OR, XOR, NOT)에 대한 데이터셋 생성"""
        data = []
        
        # AND
        and_truth = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1
        }
        for (x, y), out in and_truth.items():
            inp = [x, y] + self.gate_types['AND']
            data.append((inp, out))
        
        # OR
        or_truth = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1
        }
        for (x, y), out in or_truth.items():
            inp = [x, y] + self.gate_types['OR']
            data.append((inp, out))
        
        # XOR
        xor_truth = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0
        }
        for (x, y), out in xor_truth.items():
            inp = [x, y] + self.gate_types['XOR']
            data.append((inp, out))
        
        # NOT: 단항 연산이므로 두 번째 입력 자리는 0으로 채움
        not_truth = {
            (0,): 1,
            (1,): 0
        }
        for (x,), out in not_truth.items():
            inp = [x, 0] + self.gate_types['NOT']
            data.append((inp, out))
        
        # Tensor로 변환
        inputs = torch.tensor([d[0] for d in data], dtype=torch.float32)
        labels = torch.tensor([[d[1]] for d in data], dtype=torch.float32)
        return inputs, labels

    def build_model(self):
        """6차원 입력(논리값 2 + 게이트 one-hot 4)을 받아 처리하는 모델 생성"""
        model = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        return model

    def train(self):
        """모델 학습"""
        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.labels)
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 100 == 0:
                with torch.no_grad():
                    predicted = (outputs > 0.5).float()
                    accuracy = (predicted == self.labels).float().mean()
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
                
    def evaluate(self):
        """모델 평가 및 상세 결과 출력"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.inputs)
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == self.labels).float().mean()
            print("\nFinal Evaluation:")
            print("Detailed Predictions:")
            for inp, label, raw, pred in zip(self.inputs, self.labels, outputs, predicted):
                x1, x2 = inp[0].item(), inp[1].item()
                gate_onehot = inp[2:6]
                
                if torch.allclose(gate_onehot, torch.tensor(self.gate_types['AND'], dtype=torch.float32)):
                    gate = "AND"
                elif torch.allclose(gate_onehot, torch.tensor(self.gate_types['OR'], dtype=torch.float32)):
                    gate = "OR"
                elif torch.allclose(gate_onehot, torch.tensor(self.gate_types['XOR'], dtype=torch.float32)):
                    gate = "XOR"
                elif torch.allclose(gate_onehot, torch.tensor(self.gate_types['NOT'], dtype=torch.float32)):
                    gate = "NOT"
                else:
                    gate = "Unknown"
                
                print(f"Gate: {gate:4s} | Input: [{x1:.0f}, {x2:.0f}] | Label: {label.item():.0f} | Raw: {raw.item():.4f} | Binary: {pred.item():.0f}")
            print(f"Accuracy: {accuracy.item():.4f}")

    def predict(self, input_data):
        """
        예측 함수:
         - input_data는 리스트 형식으로 [x, y, gate_onehot(4차원)] 형태여야 함.
         - 단, NOT의 경우 두 번째 입력은 0으로 채워야 함.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            output = self.model(input_tensor)
            prediction = (output > 0.5).float()
        return prediction