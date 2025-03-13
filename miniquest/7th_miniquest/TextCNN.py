import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# 하이퍼파라미터 설정
max_features = 20000  # 사용할 단어(토큰)의 최대 개수
maxlen = 400          # 하나의 리뷰에서 사용할 단어 수(길이)
batch_size = 64
embedding_dim = 128
num_filters = 100     # 각 Conv 필터에서 추출할 특성맵 수
kernel_sizes = [3, 4, 5]  # TextCNN에서 사용할 커널(윈도우) 크기
dropout_rate = 0.5
num_epochs = 5

# IMDB 데이터 불러오기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 시퀀스 패딩 (길이가 maxlen이 되도록 자르거나 0으로 채움)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 넘파이 → 토치 텐서로 변환
x_train_t = torch.LongTensor(x_train)
y_train_t = torch.LongTensor(y_train)
x_test_t = torch.LongTensor(x_test)
y_test_t = torch.LongTensor(y_test)

# TensorDataset, DataLoader 구성
train_dataset = TensorDataset(x_train_t, y_train_t)
test_dataset = TensorDataset(x_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout_rate, num_classes=1):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embed_dim))
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)  # 최종 출력층

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            c_out = F.relu(conv(x))
            c_out = F.max_pool2d(c_out, (c_out.shape[2], c_out.shape[3]))
            c_out = c_out.squeeze(3).squeeze(2)
            conv_outputs.append(c_out)

        cat = torch.cat(conv_outputs, dim=1)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits



model = TextCNN(
    vocab_size=max_features,
    embed_dim=embedding_dim,
    num_filters=num_filters,
    kernel_sizes=kernel_sizes,
    dropout_rate=dropout_rate,
    num_classes=1  
)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)




for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x).squeeze(1)
        loss = criterion(logits, batch_y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



model.eval()
y_true = []
y_pred_probs_list = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        logits = model(batch_x).squeeze(1)
        prob = torch.sigmoid(logits)
        
        y_pred_probs_list.append(prob.cpu().numpy())
        y_true.append(batch_y.cpu().numpy())


y_true = np.concatenate(y_true, axis=0)
y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
y_pred = (y_pred_probs > 0.5).astype(np.int32)



cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# ROC-AUC 점수
roc_auc = roc_auc_score(y_true, y_pred_probs)
print(f'ROC-AUC Score: {roc_auc:.4f}')


sample_index = 0
sample_review = x_test_t[sample_index].unsqueeze(0).to(device)  
model.eval()
with torch.no_grad():
    logit = model(sample_review).squeeze(1)  
    prob = torch.sigmoid(logit).item()

print(f'샘플 리뷰의 예측 확률: {prob:.4f}')
print(f'예측 결과: {"Positive" if prob > 0.5 else "Negative"}')