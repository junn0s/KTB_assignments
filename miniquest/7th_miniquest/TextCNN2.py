import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset  # Hugging Face datasets 라이브러리 사용
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
MAX_LENGTH = 400  # 리뷰 시퀀스 최대 길이
vocab_size = 20000  # 최대 어휘 사전 크기
embed_dim = 128     # 임베딩 차원
num_classes = 2     # IMDB 이진 분류

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 1. IMDB 데이터셋 불러오기 (이미 train/test 분할되어 있음)
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]



# 2. 단어 집합(Vocabulary) 만들기
counter = Counter()
for sample in train_dataset:
    # 공백 기준 간단한 토큰화 예시 (실제 사용 시 nltk, spacy, BERT tokenizer 등을 고려)
    tokens = sample["text"].split()
    counter.update(tokens)

# 가장 많이 등장하는 vocab_size개의 단어만 추출
vocab = counter.most_common(vocab_size - 2)  # 특수 토큰([PAD], [UNK]) 고려
word2idx = {"[PAD]": 0, "[UNK]": 1}  # 0: 패딩, 1:UNK
idx = 2
for word, _ in vocab:
    word2idx[word] = idx
    idx += 1

# 단어 -> 인덱스 변환 함수
def text_to_sequence(text):
    tokens = text.split()
    sequence = []
    for token in tokens:
        if token in word2idx:
            sequence.append(word2idx[token])
        else:
            sequence.append(word2idx["[UNK]"])
    # 길이 제한
    if len(sequence) > MAX_LENGTH:
        sequence = sequence[:MAX_LENGTH]
    return sequence

# 3. Dataset, DataLoader 만들기
class IMDBDataset(Dataset):
    def __init__(self, split_dataset, word2idx):
        self.data = split_dataset
        self.word2idx = word2idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        sequence = text_to_sequence(text)
        return sequence, label

def collate_fn(batch):
    """동적 길이의 텍스트를 하나의 배치로 합치는 함수(padding)."""
    sequences, labels = zip(*batch)
    # 시퀀스 길이 측정
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    # PAD 토큰(인덱스 0)으로 패딩
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [0]*(max_len - len(seq))
        padded_sequences.append(padded_seq)
    
    return torch.LongTensor(padded_sequences), torch.LongTensor(labels)

train_data = IMDBDataset(train_dataset, word2idx)
test_data = IMDBDataset(test_dataset, word2idx)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 4. TextCNN 모델 구현
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 합성곱 레이어 여러 개 병렬 사용
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        
        conved = []
        for conv in self.convs:
            # conv -> ReLU -> MaxPooling
            # conv_out: (batch_size, num_filters, seq_len-k+1, 1)
            c = F.relu(conv(embedded)).squeeze(3)
            # 글로벌 맥스 풀링: (batch_size, num_filters)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conved.append(c)
        
        # 여러 커널 결과를 concat
        cat = torch.cat(conved, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        cat = self.dropout(cat)
        out = self.fc(cat)
        return out

# 모델 인스턴스 생성
model = TextCNN(vocab_size=len(word2idx), embed_dim=embed_dim, num_classes=num_classes).to(DEVICE)

# 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ========== 학습 루프(간단 예시) ==========
print("Training Start...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss: {avg_loss:.4f}")

print("Training Finished.")

model.eval()
y_true = []
y_predicted = []
y_pred_probs_list = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        preds = model(batch_x)  # shape: (batch_size, 2)

        # softmax로 "Positive(1)" 클래스에 대한 확률만 추출
        prob = F.softmax(preds, dim=1)[:, 1]  # (batch_size,)
        pred_label = torch.argmax(preds, dim=1)  # (batch_size,)

        y_true.extend(batch_y.cpu().tolist())         # 실제 라벨
        y_predicted.extend(pred_label.cpu().tolist()) # 예측 라벨
        y_pred_probs_list.extend(prob.cpu().tolist()) # 예측 확률(Positive)

# ------------------------------------------------------
# 혼동 행렬 시각화
# ------------------------------------------------------
cm = confusion_matrix(y_true, y_predicted)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ------------------------------------------------------
# 분류 리포트 & ROC-AUC
# ------------------------------------------------------
print("Classification Report:\n", classification_report(y_true, y_predicted, digits=4))

roc_auc = roc_auc_score(y_true, y_pred_probs_list)
print(f'ROC-AUC Score: {roc_auc:.4f}')

# ------------------------------------------------------
# (추가) 임의의 샘플 리뷰 예측
# ------------------------------------------------------
sample_index = 0
sample_x, sample_y = test_data[sample_index]  # (sequence, label)
# sequence는 이미 int 인덱스 리스트 형태
sample_x = torch.LongTensor(sample_x).unsqueeze(0).to(DEVICE)  # (1, seq_len)
model.eval()
with torch.no_grad():
    logit = model(sample_x)  # shape: (1, 2)
    prob_pos = F.softmax(logit, dim=1)[:, 1].item()  # Positive(1) 확률
print(f'[샘플 index={sample_index}] 실제 라벨: {sample_y} (0=Negative, 1=Positive)')
print(f'예측 결과: {"Positive" if prob_pos > 0.5 else "Negative"}')