import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque

# ------------------------------
# 1. 미로 데이터 생성 함수 (재귀적 백트래킹 + BFS 유효성 검증)
# ------------------------------

def generate_maze(width, height):
    """
    width, height는 홀수여야 함 (예: 28x28).
    재귀적 백트래킹을 통해 미로를 생성.
    """
    maze = np.ones((height, width), dtype=np.uint8)
    maze[1, 1] = 0  # 시작점

    visited = np.zeros(((height//2), (width//2)), dtype=bool)

    def carve_passages(cx, cy):
        visited[cy, cx] = True
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

def is_solvable(maze):
    """
    BFS를 통해 (1,1) -> (height-2, width-2)로 가는 경로가 있는지 검사.
    0: 통로, 1: 벽
    """
    h, w = maze.shape
    start = (1, 1)
    end = (h-2, w-2)
    
    if maze[start] == 1 or maze[end] == 1:
        return False
    
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([start])
    visited[start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == end:
            return True
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < h and 0 <= ny < w:
                if maze[nx, ny] == 0 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
    return False

def create_maze_dataset(n_samples=5000, img_size=28):
    """
    n_samples만큼 미로 생성 후,
    BFS로 실제로 풀 수 있는 미로만 추려서 [-1,1] 정규화.
    """
    dataset = []
    count = 0
    max_attempts = n_samples * 2  # 여유롭게 시도

    while len(dataset) < n_samples and count < max_attempts:
        maze = generate_maze(img_size, img_size)
        if is_solvable(maze):
            # 0 -> -1, 1 -> 1로 매핑
            maze = maze.astype(np.float32)
            maze = (maze - 0.5) * 2.0
            dataset.append(maze)
        count += 1

    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, axis=1)  # (N, 1, H, W)
    return dataset

# ------------------------------
# 2. PyTorch Dataset & Dataloader
# ------------------------------
class MazeDataset(data.Dataset):
    def __init__(self, np_data):
        self.data = torch.tensor(np_data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------------
# 3. GAN 모델 구성 (Generator & Discriminator)
# ------------------------------
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    """
    입력: latent vector (z) -> 출력: 28x28 미로 이미지
    채널 수를 조금 늘려서 학습 안정성을 높임.
    """
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 7*7*256),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(0.2, inplace=True),

            View((-1, 256, 7, 7)),  # (batch, 256, 7, 7)

            # upsample to 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # upsample to 28x28
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1,1]
        )
        
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """
    입력: 28x28 미로 이미지 -> 출력: 진짜/가짜 판별
    """
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Flatten(),
            nn.Linear(256 * (img_size // 8) * (img_size // 8), 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.net(img)

# ------------------------------
# 4. 학습 준비
# ------------------------------

def build_models(latent_dim=100, img_size=28):
    generator = Generator(latent_dim)
    discriminator = Discriminator(img_size)
    return generator, discriminator

def init_weights(m):
    """
    가중치 초기화 (DCGAN 스타일 권장 초기화)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ------------------------------
# 5. 학습 루프 (GAN)
# ------------------------------

def train_gan(
    generator, 
    discriminator, 
    dataloader, 
    device, 
    latent_dim=100, 
    lr=0.0002, 
    beta1=0.5, 
    num_epochs=5000,
    sample_interval=500
):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    generator.to(device)
    discriminator.to(device)

    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size_current = real_imgs.size(0)

            # ---------------------------
            # 1) Discriminator 학습
            # ---------------------------
            optimizer_D.zero_grad()

            real_labels = torch.ones((batch_size_current, 1), device=device)
            fake_labels = torch.zeros((batch_size_current, 1), device=device)

            # 진짜 이미지 판별
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # 가짜 이미지 판별
            noise = torch.randn(batch_size_current, latent_dim, device=device)
            fake_imgs = generator(noise)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ---------------------------
            # 2) Generator 학습
            # ---------------------------
            optimizer_G.zero_grad()

            # Generator는 가짜 이미지를 진짜로 판별받길 원함
            output = discriminator(fake_imgs)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        end_time = time.time()
        if epoch % sample_interval == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {d_loss.item():.4f} | Loss_G: {g_loss.item():.4f} | Time: {end_time - start_time:.2f} sec")
            save_sample_images(generator, device, epoch, latent_dim)

    return generator, discriminator

def save_sample_images(model, device, epoch, latent_dim=100, examples=4):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(examples, latent_dim, device=device)
        generated_imgs = model(noise)
        generated_imgs = (generated_imgs + 1) / 2.0  # [-1,1] -> [0,1]
        grid = vutils.make_grid(generated_imgs, nrow=examples, normalize=True)
        np_grid = grid.cpu().numpy().transpose((1, 2, 0))
        plt.figure(figsize=(examples*3, 3))
        plt.imshow(np_grid, cmap='gray')
        plt.title(f'Epoch {epoch}')
        plt.axis('off')
        plt.show()
    model.train()

# ------------------------------
# 6. 전체 실행
# ------------------------------

if __name__ == "__main__":
    # 1) 데이터셋 생성
    img_size = 28
    n_samples = 5000
    print("미로 데이터 생성 중... (유효성 검증 포함)")
    maze_data = create_maze_dataset(n_samples, img_size)
    print(f"Dataset shape: {maze_data.shape}")

    dataset = MazeDataset(maze_data)
    batch_size = 32
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2) 모델 초기화
    latent_dim = 100
    generator, discriminator = build_models(latent_dim, img_size)

    # 가중치 초기화
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # 3) 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5000
    sample_interval = 500

    print("GAN 학습 시작...")
    generator, discriminator = train_gan(
        generator, 
        discriminator, 
        dataloader, 
        device,
        latent_dim=latent_dim, 
        num_epochs=num_epochs,
        sample_interval=sample_interval
    )

    # 4) 최종 샘플 출력
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(9, latent_dim, device=device)
        gen_imgs = generator(noise)
        gen_imgs = (gen_imgs + 1) / 2.0  # [0,1]

    grid = vutils.make_grid(gen_imgs, nrow=3, normalize=True)
    np_grid = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(6, 6))
    plt.imshow(np_grid, cmap='gray')
    plt.axis('off')
    plt.title("Final Generated Maze Samples")
    plt.show()