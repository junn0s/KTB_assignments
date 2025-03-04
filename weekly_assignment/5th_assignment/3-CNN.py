import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


### 원 및 사각형 생성, 0과 1로 라벨링 ###

class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=28, transform=None):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.data, self.labels = self._generate_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = np.expand_dims(img, axis=0)  # (1, 28, 28)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)
            
        return img, label
    
    def _generate_data(self):
        data_list = []
        label_list = []
        for _ in range(self.num_samples):
            shape_type = np.random.choice([0, 1])  
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            radius = np.random.randint(self.image_size//6, self.image_size//3)
            center_x = np.random.randint(radius, self.image_size - radius)
            center_y = np.random.randint(radius, self.image_size - radius)
            
            if shape_type == 0:
                Y, X = np.ogrid[:self.image_size, :self.image_size]
                mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
                img[mask] = 1.0

            else:
                half_len = radius
                top    = max(0, center_y - half_len)
                bottom = min(self.image_size, center_y + half_len)
                left   = max(0, center_x - half_len)
                right  = min(self.image_size, center_x + half_len)
                img[top:bottom, left:right] = 1.0
            
            data_list.append(img)
            label_list.append(shape_type)
        
        return np.array(data_list), np.array(label_list)
    



dataset = ShapeDataset(num_samples=2000, image_size=28)

train_size = int(len(dataset)*0.8)
test_size  = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # -> (batch, 8, 28, 28)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # -> (batch, 16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                           # -> size half
        self.fc1  = nn.Linear(16 * 7 * 7, 32)
        self.fc2  = nn.Linear(32, 2)  # circle(0), square(1)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))    # (8, 28, 28)
        x = self.pool(x)                         # (8, 14, 14)
        x = nn.functional.relu(self.conv2(x))    # (16, 14, 14)
        x = self.pool(x)                         # (16, 7, 7)
        
        x = x.view(x.size(0), -1)   # (batch, 16*7*7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    
    acc = correct / total
    print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Test Acc: {acc:.4f}")


### 시각화 ###


for i in range(10):
    img, label = dataset[i]
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Label: {label} (0=circle,1=square)")
    plt.axis('off')
    plt.show()