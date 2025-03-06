import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
num_epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10)         
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(x)              
        x = self.relu(self.conv2(x))   
        x = self.pool(x)               
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x) 
        return x

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%\n")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

final_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {final_accuracy:.2f}%")



######################### 시각화 #########################


import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

dataiter = iter(test_loader)
images, labels = next(dataiter)

num_samples = 8

fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3))
for i in range(num_samples):
    sample_img = images[i]
    true_label = labels[i].item()
    
    model.eval()
    with torch.no_grad():
        output = model(sample_img.unsqueeze(0))  
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()  

    predicted_class = int(np.argmax(probabilities))
    
    ax_img = axes[i, 0]
    sample_img_cpu = sample_img.cpu().squeeze()
    sample_img_orig = sample_img_cpu * 0.3081 + 0.1307  
    ax_img.imshow(sample_img_orig, cmap='gray')
    ax_img.set_title(f"True: {true_label}, Pred: {predicted_class}")
    ax_img.axis('off')
    
    ax_bar = axes[i, 1]
    ax_bar.bar(range(10), probabilities)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel("Digit")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title("Prediction Probabilities")

plt.tight_layout()
plt.show()