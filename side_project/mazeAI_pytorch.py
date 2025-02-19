import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ------------------------------
# 1. 미로 데이터 생성 함수 (재귀적 백트래킹)
# ------------------------------
def generate_maze(width, height):
    # width, height는 홀수여야 함 (예: 28x28)
    maze = np.ones((height, width), dtype=np.uint8)
    maze[1, 1] = 0  # 시작점
    
    visited = np.zeros(((height//2), (width//2)), dtype=bool)
    
    def carve_passages(cx, cy):
        visited[cy, cx] = True
        # 상하좌우 방향 (dx, dy)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width//2 and 0 <= ny < height//2 and not visited[ny, nx]:
                # 현재 셀과 다음 셀 사이의 벽 제거
                maze[cy*2 + dy, cx*2 + dx] = 0
                maze[ny*2, nx*2] = 0
                carve_passages(nx, ny)
    
    carve_passages(0, 0)
    return maze

def create_maze_dataset(n_samples=1000, img_size=28):
    dataset = []
    for _ in range(n_samples):
        maze = generate_maze(img_size, img_size)
        # float32로 변환 후 [-1, 1] 정규화 (0 -> -1, 1 -> 1)
        maze = maze.astype(np.float32)
        maze = (maze - 0.5) * 2.0
        dataset.append(maze)
    dataset = np.array(dataset)
    # (N, H, W) -> (N, 1, H, W)
    dataset = np.expand_dims(dataset, axis=1)
    return dataset

# 데이터셋 생성
img_size = 28
n_samples = 1000
maze_data = create_maze_dataset(n_samples, img_size)
print("Dataset shape:", maze_data.shape)

# PyTorch Dataset 정의
class MazeDataset(data.Dataset):
    def __init__(self, np_data):
        self.data = torch.tensor(np_data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MazeDataset(maze_data)
batch_size = 32
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ------------------------------
# 2. GAN 모델 구성 (Generator & Discriminator)
# ------------------------------

latent_dim = 100

# Generator: latent vector -> 28x28 미로 이미지
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.LeakyReLU(0.2, inplace=True),
            # reshape: (batch, 128, 7, 7)
            View((-1, 128, 7, 7)),
            # upsample to 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # upsample to 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 출력 범위 [-1, 1]
        )
        
    def forward(self, z):
        return self.net(z)

# 간단한 helper 모듈: tensor의 shape 변경 (reshape)
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape  # shape 튜플 (예: (-1, 128, 7, 7))
        
    def forward(self, x):
        return x.view(*self.shape)

# Discriminator: 28x28 미로 이미지 -> 진짜/가짜 판별
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Flatten(),
            nn.Linear(128 * (img_shape[0] // 4) * (img_shape[1] // 4), 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.net(img)

img_shape = (img_size, img_size)
generator = Generator(latent_dim)
discriminator = Discriminator(img_shape)

print(generator)
print(discriminator)

# ------------------------------
# 3. GAN 학습 준비
# ------------------------------
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# ------------------------------
# 4. GAN 학습 루프
# ------------------------------
num_epochs = 5000
sample_interval = 500  # 몇 에폭마다 결과 출력

def save_sample_images(epoch, examples=4):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(examples, latent_dim, device=device)
        generated_imgs = generator(noise)
        # [0,1] 범위로 스케일 변환
        generated_imgs = (generated_imgs + 1) / 2.0
        grid = vutils.make_grid(generated_imgs, nrow=examples, normalize=True)
        np_grid = grid.cpu().numpy().transpose((1, 2, 0))
        plt.figure(figsize=(examples*3, 3))
        plt.imshow(np_grid, cmap='gray')
        plt.title(f'Epoch {epoch}')
        plt.axis('off')
        plt.show()
    generator.train()

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size_current = real_imgs.size(0)
        
        # 진짜, 가짜 라벨 생성
        real_labels = torch.ones((batch_size_current, 1), device=device)
        fake_labels = torch.zeros((batch_size_current, 1), device=device)
        
        # ---------------------------
        # Discriminator 학습
        # ---------------------------
        optimizer_D.zero_grad()
        # 진짜 이미지에 대한 판별
        output_real = discriminator(real_imgs)
        loss_real = criterion(output_real, real_labels)
        
        # Generator에서 생성한 가짜 이미지에 대한 판별
        noise = torch.randn(batch_size_current, latent_dim, device=device)
        fake_imgs = generator(noise)
        output_fake = discriminator(fake_imgs.detach())
        loss_fake = criterion(output_fake, fake_labels)
        
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
        
        # ---------------------------
        # Generator 학습
        # ---------------------------
        optimizer_G.zero_grad()
        # Generator는 가짜 이미지를 진짜로 판별받길 원함
        output = discriminator(fake_imgs)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizer_G.step()
    
    end_time = time.time()
    if epoch % sample_interval == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | Time: {end_time - start_time:.2f} sec")
        save_sample_images(epoch)

# ------------------------------
# 5. 학습 완료 후 생성 결과 확인
# ------------------------------
generator.eval()
with torch.no_grad():
    noise = torch.randn(9, latent_dim, device=device)
    gen_imgs = generator(noise)
    gen_imgs = (gen_imgs + 1) / 2.0  # [0,1] 스케일 변환

grid = vutils.make_grid(gen_imgs, nrow=3, normalize=True)
np_grid = grid.cpu().numpy().transpose((1, 2, 0))

plt.figure(figsize=(6, 6))
plt.imshow(np_grid, cmap='gray')
plt.axis('off')
plt.title("Generated Maze Samples")
plt.show()


# ------------------------------
# 6. 모델 저장 및 불러오기
# ------------------------------
# 학습 루프가 끝난 후 또는 원하는 시점에 체크포인트 저장

# 저장할 때 포함할 정보: 현재 에폭, Generator/Discriminator의 state_dict, Optimizer의 state_dict, 최근 손실 값 등
checkpoint = {
    'epoch': epoch,  # 마지막 에폭 번호
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'loss_G': loss_G.item(),  # 마지막 Generator 손실 값
    'loss_D': loss_D.item(),  # 마지막 Discriminator 손실 값
}

# 체크포인트를 파일로 저장 (예: 'gan_checkpoint.pth')
torch.save(checkpoint, 'gan_checkpoint.pth')
print("모델 체크포인트 저장 완료.")

# 저장한 체크포인트를 불러와서 모델과 옵티마이저의 상태를 복원합니다.
# (이 코드는 새로운 세션 또는 동일한 코드 실행 시 사용할 수 있습니다.)

# 저장된 체크포인트 로드
checkpoint = torch.load('gan_checkpoint.pth', map_location=device)

# Generator와 Discriminator 객체를 동일한 클래스 구조로 다시 생성한 후,
# 저장된 state_dict를 불러옵니다.
generator = Generator(latent_dim).to(device)
discriminator = Discriminator(img_shape).to(device)

generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# 옵티마이저도 동일하게 초기화한 후 state_dict를 복원합니다.
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

# 이어서 학습을 계속할 때, 마지막 에폭 번호 이후부터 진행할 수 있습니다.
start_epoch = checkpoint['epoch'] + 1

print(f"체크포인트에서 불러오기 완료. 이어서 {start_epoch} 에폭부터 학습을 진행할 수 있습니다.")