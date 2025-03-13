import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text.lower()  # 입력 텍스트 저장
        self.sequence_length = sequence_length  # 시퀀스 길이 저장

        # 텍스트에 있는 고유한 문자를 인덱스로 변환하는 딕셔너리 생성
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        # 인덱스를 문자로 변환하는 딕셔너리 생성
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        # 텍스트 데이터를 인덱스 리스트로 변환
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        # 데이터셋의 길이 반환 (전체 데이터 길이에서 시퀀스 길이를 뺀 값)
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # 주어진 인덱스에서 시퀀스를 반환
        return (
            # 입력 시퀀스: idx부터 idx+sequence_length까지의 데이터
            torch.tensor(self.data[idx:idx+self.sequence_length], dtype=torch.long),
            # 타겟 시퀀스: idx+1부터 idx+sequence_length+1까지의 데이터
            torch.tensor(self.data[idx+1:idx+self.sequence_length+1], dtype=torch.long)
        )
    
text = "The quick brown fox jumps over the lazy dog. This is a sample text for the RNN model. The purpose is to generate text based on the learned patterns."
sequence_length = 30  # 시퀀스 길이
dataset = TextDataset(text, sequence_length)  # TextDataset 클래스의 인스턴스 생성

# PyTorch DataLoader 생성
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 단어 임베딩 레이어
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # RNN 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)  # 완전 연결(전결합) 레이어

    def forward(self, x):
        x = self.embedding(x)  # 입력 텐서를 임베딩 벡터로 변환
        out, _ = self.rnn(x)  # RNN 레이어를 통과하여 은닉 상태를 계산
        out = self.fc(out)  # RNN 출력 텐서를 전결합 레이어로 변환하여 최종 출력 생성
        return out

# 하이퍼파라미터 설정
vocab_size = len(dataset.char_to_idx)  # 단어 집합의 크기, dataset.char_to_idx의 길이로 설정
embed_dim = 20  # 임베딩 차원
hidden_dim = 128  # RNN 은닉 상태 차원
output_dim = vocab_size  # 출력 차원, 예측할 단어의 수

# 모델 인스턴스 생성
model = SimpleRNN(vocab_size, embed_dim, hidden_dim, output_dim)
# 설정된 하이퍼파라미터를 사용하여 SimpleRNN 클래스의 인스턴스를 생성


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 여부 확인 및 장치 설정
model = model.to(device)  # 모델을 선택된 장치로 이동

num_epochs = 50  # 학습 에포크 수 설정

for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0  # 에포크 동안의 손실 누적을 위한 변수 초기화

    for inputs, targets in dataloader:  # 데이터로더를 통해 배치 단위로 데이터 불러오기
        inputs, targets = inputs.to(device), targets.to(device)  # 입력과 타겟 데이터를 장치로 이동
        outputs = model(inputs)  # 모델을 통해 출력 계산
        outputs = outputs.view(-1, outputs.size(-1))  # 출력을 2차원 텐서로 변환
        targets = targets.view(-1)  # 타겟을 1차원 텐서로 변환

        loss = criterion(outputs, targets)  # 손실 계산

        optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
        loss.backward()  # 역전파를 통해 그래디언트 계산
        optimizer.step()  # 옵티마이저를 통해 파라미터 업데이트

        running_loss += loss.item() * inputs.size(0)  # 손실 값을 누적

    epoch_loss = running_loss / len(dataset)  # 에포크 동안의 평균 손실 계산
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')  # 에포크와 손실 출력


def evaluate(model, dataloader):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0  # 총 손실 초기화
    total_correct = 0  # 총 정확한 예측 수 초기화
    total_samples = 0  # 총 샘플 수 초기화

    with torch.no_grad():  # 평가 시에는 그래디언트를 계산하지 않음
        for inputs, targets in dataloader:  # 데이터로더에서 배치 단위로 데이터 가져오기
            inputs, targets = inputs.to(device), targets.to(device)  # 입력과 타겟을 장치로 이동
            outputs = model(inputs)  # 모델을 사용하여 예측 수행
            outputs = outputs.view(-1, outputs.size(-1))  # 출력 텐서의 형태를 변경
            targets = targets.view(-1)  # 타겟 텐서의 형태를 변경

            loss = criterion(outputs, targets)  # 손실 계산
            total_loss += loss.item() * targets.size(0)  # batch_size * sequence_length

            _, predicted = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스 예측
            total_correct += (predicted == targets).sum().item()  # 정확한 예측 수 추가
            total_samples += targets.size(0)  # 총 샘플 수 추가

    average_loss = total_loss / total_samples  # 평균 손실 계산
    accuracy = total_correct / total_samples * 100  # 정확도 계산
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')  # 결과 출력

# 모델과 데이터로더를 사용하여 평가 함수 호출
evaluate(model, dataloader)


def generate_text(model, start_text, length, temperature=0.8):
    model.eval()  
    chars = [dataset.char_to_idx[ch] for ch in start_text if ch in dataset.char_to_idx]

    # 시작 문자가 너무 짧으면 패딩 추가
    if len(chars) < sequence_length:
        chars = [dataset.char_to_idx[' ']] * (sequence_length - len(chars)) + chars

    input_seq = torch.tensor(chars[-sequence_length:], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_text  

    with torch.no_grad():  
        for _ in range(length):
            output = model(input_seq)
            
            # 샘플링 방식 적용
            output = output[:, -1, :] / temperature  
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            predicted = torch.multinomial(probabilities, 1).item()
            
            next_char = dataset.idx_to_char[predicted]  
            generated_text += next_char  
            
            # 슬라이딩 윈도우 방식으로 입력 업데이트
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted]], device=device)], dim=1)

    return generated_text

start_text = "hello "  # 텍스트 생성을 시작할 시드 문자열
generated_text = generate_text(model, start_text, 100)  # 모델을 사용하여 20개의 문자를 생성
print(f'Generated Text: {generated_text}')  # 생성된 텍스트 출력