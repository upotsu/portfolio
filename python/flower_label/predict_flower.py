import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# モデルのロード（絶対パスで指定）
#model_path = r"ファイルの保存先//flower_model.h5"
model = tf.keras.models.load_model(model_path)

# クラスラベルの設定
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 判定するフォルダのパス
#folder_path = r"保存先\img_path"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # 学習時と同じ正規化

        preds = model.predict(img_array, verbose=0)
        top_idx = np.argmax(preds)
        print(f"{filename} → {class_labels[top_idx]} ({preds[0][top_idx]*100:.2f}%)")
