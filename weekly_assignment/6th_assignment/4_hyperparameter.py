import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

torch.backends.cudnn.benchmark = True  # 고정 크기 입력 시 최적 커널 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resize_size = 64  

transform_train = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)





num_classes = 10

def create_model(learning_rate=0.001):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [256, 512, 1024]  
epochs = 10

best_accuracy = 0.0
best_params = {}

scaler = torch.cuda.amp.GradScaler()  # AMP 스케일러 추가

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\n--- Training with Learning Rate: {lr}, Batch Size: {batch_size}, Image Size: {resize_size}x{resize_size} ---")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model, optimizer, criterion = create_model(learning_rate=lr)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():  
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

        # 테스트셋 평가
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
        print(f"Test Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'learning_rate': lr, 'batch_size': batch_size}

        print("-" * 50)

print("\n=== Hyperparameter Search Result ===")
print("Best Parameters:", best_params)
print(f"Best Test Accuracy: {best_accuracy:.4f}")


# 최적 파라미터로 모델 재학습
print("\n=== Re-training model with best hyperparameters ===")

best_lr = best_params['learning_rate']
best_batch_size = best_params['batch_size']

train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, num_workers=4, pin_memory=True)

model, optimizer, criterion = create_model(learning_rate=best_lr)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # AMP 적용
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

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

final_accuracy = correct / total
print(f"\nFinal Test Accuracy with Image Size {resize_size}x{resize_size}: {final_accuracy:.4f}")