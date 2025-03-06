import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt



batch_size_resnet = 256
batch_size_vgg = 256  
num_epochs_resnet = 10
num_epochs_vgg = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader_resnet = DataLoader(train_dataset, batch_size=batch_size_resnet, 
                                 shuffle=True, num_workers=4, pin_memory=True)
test_loader_resnet = DataLoader(test_dataset, batch_size=batch_size_resnet, 
                                shuffle=False, num_workers=4, pin_memory=True)

train_loader_vgg = DataLoader(train_dataset, batch_size=batch_size_vgg, 
                              shuffle=True, num_workers=4, pin_memory=True)
test_loader_vgg = DataLoader(test_dataset, batch_size=batch_size_vgg, 
                             shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# 모델 정의
def build_resnet50():
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def build_vgg16():
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
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, 10)
    )
    return model


# 학습, 평가
def train_and_evaluate(model, train_loader, test_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)

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

        # 에폭별로 테스트 정확도 계산
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

        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")
    
    return train_losses, test_accuracies


# resnet, vgg 학습
print("===== ResNet50 Training =====")
resnet50 = build_resnet50()
resnet_num_epochs = num_epochs_resnet
resnet_lr = 0.0004  
resnet_train_losses, resnet_test_accuracies = train_and_evaluate(
    resnet50, train_loader_resnet, test_loader_resnet, 
    resnet_num_epochs, resnet_lr, device
)

print("\n===== VGG16 Training =====")
vgg16 = build_vgg16()
vgg_num_epochs = num_epochs_vgg
vgg_lr = 0.001  
vgg_train_losses, vgg_test_accuracies = train_and_evaluate(
    vgg16, train_loader_vgg, test_loader_vgg, 
    vgg_num_epochs, vgg_lr, device
)




# 결과 시각화
plt.figure()
plt.plot(range(1, resnet_num_epochs+1), resnet_train_losses, label='ResNet50 Loss')
plt.plot(range(1, vgg_num_epochs+1), vgg_train_losses, label='VGG16 Loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, resnet_num_epochs+1), resnet_test_accuracies, label='ResNet50 Acc')
plt.plot(range(1, vgg_num_epochs+1), vgg_test_accuracies, label='VGG16 Acc')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.show()