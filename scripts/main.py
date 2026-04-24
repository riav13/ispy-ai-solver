import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

train_dir = "data/train"
test_dir = "data/test"

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise" ]
label_encode = {label: i for i, label in enumerate(emotion_labels)}

def load_images_from_folder(base_folder, label_names):
    X = []
    y = []

    for label in label_names:
        folder_path = os.path.join(base_folder, label)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            try:
                img = Image.open(file_path).convert("L")
                img_array = np.array(img, dtype=np.float32)

                X.append(img_array)
                y.append(label_encode[label])

            except Exception as e:
                print(f"Could not load {file_path}: {e}")

    return np.array(X), np.array(y)