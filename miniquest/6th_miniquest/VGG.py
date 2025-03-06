import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
from torchvision import models

num_classes = 10
input_shape = (3, 224, 224)

batch_size = 256
num_classes = 10
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


model = models.vgg16(weights=None)
original_features = model.features
modified_features = nn.Sequential()
modified_features.add_module('0', nn.Conv2d(1, 64, kernel_size=3, padding=1))
for i in range(1, len(original_features)):
    modified_features.add_module(str(i), original_features[i])

model.features = modified_features
model.features = nn.Sequential(*list(model.features)[:-1])
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.classifier = nn.Sequential(
    nn.Linear(512, 256),  # VGG16 마지막 컨볼루션 레이어는 512 채널
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(256, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')