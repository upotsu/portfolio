import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# モデルのロード
#model_path = r"学習用モデルデータの保存先/flower_model.h5"
model = tf.keras.models.load_model(model_path)

# クラスラベル
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# GUI作成
root = tk.Tk()
root.title("Flower Classifier")
root.geometry("500x600")

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # 画像表示
    pil_img = Image.open(file_path)
    pil_img.thumbnail((400, 400))
    tk_img = ImageTk.PhotoImage(pil_img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    # 画像判定
    img_array = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array, verbose=0)
    top_idx = np.argmax(preds)
    result_label.config(text=f"{class_labels[top_idx]} ({preds[0][top_idx]*100:.2f}%)")

# ボタン
btn = tk.Button(root, text="画像を選択して判定", command=select_image)
btn.pack(pady=20)

root.mainloop()
