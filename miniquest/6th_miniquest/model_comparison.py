import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터
batch_size = 256
num_epochs = 10
learning_rate = 0.0004

# 데이터셋 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 모델1: SimpleCNN 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 모델2: ResNet50 변경 버전
def get_resnet50():
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# 훈련 및 평가 공통 함수
def train_and_evaluate(model, train_loader, test_loader, device, num_epochs, learning_rate):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 테스트 정확도 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return train_losses, test_accuracies


# 두 모델 학습 및 평가
print("Training SimpleCNN...")
simplecnn = SimpleCNN()
simplecnn_losses, simplecnn_accuracies = train_and_evaluate(simplecnn, train_loader, test_loader, device, num_epochs, learning_rate)

print("\nTraining ResNet50...")
resnet50 = get_resnet50()
resnet50_losses, resnet50_accuracies = train_and_evaluate(resnet50, train_loader, test_loader, device, num_epochs, learning_rate)

# 성능 시각화
plt.figure(figsize=(14, 5))

# 손실 비교
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), simplecnn_losses, label='SimpleCNN', marker='o')
plt.plot(range(1, num_epochs+1), resnet50_losses, label='ResNet50', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Comparison')
plt.legend()

# 정확도 비교
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), simplecnn_accuracies, label='SimpleCNN', marker='o')
plt.plot(range(1, num_epochs+1), resnet50_accuracies, label='ResNet50', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# 최종 결과 출력
print(f"SimpleCNN Final Test Accuracy: {simplecnn_accuracies[-1]:.2f}%")
print(f"ResNet50 Final Test Accuracy: {resnet50_accuracies[-1]:.2f}%")