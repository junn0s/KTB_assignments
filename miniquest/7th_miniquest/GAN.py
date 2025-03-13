import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 128
latent_dim = 100
lr = 0.0002
epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 변환 및 로딩 최적화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Generator (CNN 기반)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # (B, 1, 28, 28) 형태로 reshape

# Discriminator (CNN 기반)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))  # (B, 784)로 펼쳐서 입력

# 모델 초기화
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 손실 함수 & 최적화 함수
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 학습 시작
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        batch_size_now = images.size(0)
        real_images = images.to(device)
        real_labels = torch.ones(batch_size_now, 1).to(device)
        fake_labels = torch.zeros(batch_size_now, 1).to(device)

        # 1. Generator에서 가짜 이미지 생성
        z = torch.randn(batch_size_now, latent_dim).to(device)
        fake_images = generator(z)

        # 2. Discriminator 학습
        discriminator.zero_grad()
        real_loss = criterion(discriminator(real_images), real_labels)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # 3. Generator 학습
        generator.zero_grad()
        g_loss = criterion(discriminator(fake_images), real_labels)  # 가짜 이미지를 진짜라고 속이도록 학습
        g_loss.backward()
        optimizer_G.step()

    # 로그 출력
    print(f"[Epoch {epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

# 결과 이미지 생성
z = torch.randn(16, latent_dim).to(device)
fake_images = generator(z).cpu().detach()
grid = make_grid(fake_images, nrow=4, normalize=True)

# 시각화
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
plt.axis('off')
plt.title("Generated MNIST Images")
plt.show()