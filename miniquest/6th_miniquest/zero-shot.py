import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in [0, 1, 2, 3, 4]]
train_dataset = Subset(full_train_dataset, train_indices)

test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label in [5, 6, 7, 8, 9]]
test_dataset = Subset(full_test_dataset, test_indices)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)





from torchvision.models import ResNet50_Weights
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

old_conv = resnet.conv1
new_conv = nn.Conv2d(1, old_conv.out_channels,
                     kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride,
                     padding=old_conv.padding,
                     bias=False)

with torch.no_grad():
    new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
resnet.conv1 = new_conv

in_features = resnet.fc.in_features
resnet.fc = nn.Identity()

for param in resnet.parameters():
    param.requires_grad = False
resnet = resnet.to(device)
resnet.eval()

classifier = nn.Linear(in_features, 5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)



# 학습 (0~4)
num_epochs = 10
print("훈련 시작 (0~4 숫자)...")
for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device)   
        labels = labels.to(device)  
        
        with torch.no_grad():
            features = resnet(imgs)  
        outputs = classifier(features)  
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


features_by_class = {i: [] for i in range(5)}
with torch.no_grad():
    for imgs, labels in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
        imgs = imgs.to(device)
        feats = resnet(imgs)  
        for feat, label in zip(feats, labels):
            features_by_class[label.item()].append(feat.cpu().numpy())

centroids = {}
for cls in features_by_class:
    feats_arr = np.array(features_by_class[cls])
    centroids[cls] = np.mean(feats_arr, axis=0)  # (2048,)

# 예측
print("\nZero-Shot 예측 (테스트 데이터: 숫자 5~9)")
zero_shot_results = []
with torch.no_grad():
    for img, true_label in test_loader:
        img = img.to(device)
        feat = resnet(img)  
        feat_np = feat.cpu().numpy().flatten()
        distances = {}
        for cls in centroids:
            centroid = centroids[cls]
            distances[cls] = np.linalg.norm(feat_np - centroid)
        # 가장 가까운 centroid와의 거리가 가장 작으면 해당 클래스로 예측
        pred_cls = min(distances, key=distances.get)
        zero_shot_results.append((true_label.item(), pred_cls, distances))



print("\n[일부 Zero-Shot 예측 결과]")
for i in range(10):
    true_label, pred_cls, dists = zero_shot_results[i]
    print(f"실제 라벨 (원래 5~9): {true_label}  ->  예측된 클래스 (0~4): {pred_cls}")
    print("각 클래스와의 거리:", {k: round(v, 3) for k, v in dists.items()})
    print("-"*40)