import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

X_train, y_train = load_images_from_folder(train_dir, emotion_labels)
X_test, y_test = load_images_from_folder(test_dir, emotion_labels)

"""print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)"""

"""print("First label number:", y_train[0])
print("Image shape:", X_train[0].shape)

plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()"""

"""for label in emotion_labels:
    print(label, np.sum(y_train == label_encode[label]))"""

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

"""print("X_train_flat shape:", X_train_flat.shape)
print("X_test_flat shape:", X_test_flat.shape)"""

pca = PCA(n_components = 100, random_state = 42)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

"""print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)"""

rf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)
rf.fit(X_train_pca, y_train)
y_pred = rf.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))