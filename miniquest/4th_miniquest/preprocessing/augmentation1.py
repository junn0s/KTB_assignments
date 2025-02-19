from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "Lenna.jpg")

# 이미지 열기
img = Image.open(img_path)

rotated_img = img.rotate(45)  # 회전
flipped_img = ImageOps.mirror(img)  # 뒤집기

tx, ty = 50, 30  # x축, y축 이동량
translated_img = img.transform(  # 이동
    img.size,
    Image.AFFINE,
    (1, 0, tx, 0, 1, ty)
)

# 확대
width, height = img.size
crop_width, crop_height = int(width * 0.8), int(height * 0.8)
left = (width - crop_width) // 2
upper = (height - crop_height) // 2
right = left + crop_width
lower = upper + crop_height
cropped_img = img.crop((left, upper, right, lower))
zoomed_img = cropped_img.resize(img.size, Image.LANCZOS)

# 색 변경
enhancer = ImageEnhance.Color(img)
color_transformed_img = enhancer.enhance(1.5)  # 1.5배 채도 증가

# 노이즈 추가
img_np = np.array(img)
noise = np.random.normal(0, 25, img_np.shape).astype(np.int16)
noisy_img_np = img_np.astype(np.int16) + noise
noisy_img_np = np.clip(noisy_img_np, 0, 255).astype(np.uint8)
noisy_img = Image.fromarray(noisy_img_np)

# 결과
rotated_img.save("rotated_img.jpg")
flipped_img.save("flipped_img.jpg")
translated_img.save("translated_img.jpg")
zoomed_img.save("zoomed_img.jpg")
color_transformed_img.save("color_transformed_img.jpg")
noisy_img.save("noisy_img.jpg")