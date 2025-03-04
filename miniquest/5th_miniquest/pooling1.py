import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input_image = np.array([
    [1, 1, 2, 4, 5, 6],
    [1, 3, 3, 2, 1, 1],
    [4, 6, 1, 1, 2, 3],
    [3, 4, 2, 1, 5, 6],
    [5, 2, 3, 4, 4, 3],
    [1, 1, 2, 2, 3, 4]
], dtype=np.float32).reshape(1, 6, 6, 1)  # (batch_size, height, width, channels)

# PyTorch 스타일로 변환: (batch, channels, height, width)
input_image_tensor = torch.tensor(input_image).permute(0, 3, 1, 2)

pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
pooled_image_tensor = pool(input_image_tensor)

input_image_np = input_image_tensor.squeeze().numpy()
pooled_image_np = pooled_image_tensor.squeeze().numpy()


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_image_np, cmap='gray', vmin=0, vmax=6)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Pooled Image")
plt.imshow(pooled_image_np, cmap='gray', vmin=0, vmax=6)
plt.axis('off')

plt.show()