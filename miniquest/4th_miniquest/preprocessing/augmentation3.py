import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# 1. 데이터셋 불러오기 및 전처리
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 2. 현재 파일의 위치를 기준으로 모델 파일 경로 설정 후, 저장된 모델 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'aug_model3.h5')
model = tf.keras.models.load_model(model_path)

# 모델이 저장 시 컴파일 상태를 유지했다면 재컴파일은 선택사항입니다.
# 재컴파일이 필요한 경우:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. 데이터 증강 설정 (이전과 동일하게)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)
train_generator = datagen.flow(train_images, train_labels, batch_size=64)

# 4. 추가로 10 에포크 더 학습
model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

# 5. 추가 학습 후 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Continued training - Test accuracy: {test_acc}')

model.save('aug_model4.h5')