import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# データジェネレータ（画像の水増し＋正規化）
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20%をバリデーションに
)

# トレーニングデータ
train_generator = datagen.flow_from_directory(
    #r'ファイルの保存先',
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='training'  # 学習用
)

# バリデーションデータ
validation_generator = datagen.flow_from_directory(
    #r'ファイルの保存先',
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'  # バリデーション用
)

# MobileNetV2をベースに転移学習
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # 最初は凍結

# モデル構築
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# コンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# モデル保存
#model.save(r'ファイルの保存先'\\flower_model.h5')
print("学習完了。モデル保存: flower_model.h5")
