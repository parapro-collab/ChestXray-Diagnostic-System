import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1️⃣ Chargement et prétraitement des images avec vérification
dataset_dir = "data/raw/images"
classes = ["normal", "pneumonia"]

X, y = [], []
for idx, label in enumerate(classes):
    folder = os.path.join(dataset_dir, label)
    if not os.path.exists(folder):
        print(f"⚠️ Dossier manquant: {folder}")
        continue
        
    for img_file in os.listdir(folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_file)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                X.append(img_array)
                y.append(idx)
            except Exception as e:
                print(f"Erreur avec {img_path}: {e}")

if len(X) == 0:
    raise ValueError("Aucune image chargée! Vérifiez le chemin du dataset.")

X = np.array(X)
y = np.array(y)

print(f"📊 Dataset chargé: {len(X)} images")
print(f"📈 Répartition des classes: {np.unique(y, return_counts=True)}")

X = preprocess_input(X)
y_cat = to_categorical(y, num_classes=2)

# Split train/test avec validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# 2️⃣ Feature extraction avec gestion de mémoire
efficient_model = EfficientNetV2S(
    weights='imagenet', 
    include_top=False, 
    pooling='avg',
    input_shape=(224, 224, 3)
)

# Extraction par lots pour économiser la mémoire
def extract_features(model, images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        features_batch = model.predict(batch, verbose=0)
        features.append(features_batch)
    return np.vstack(features)

print("⏳ Extraction des features...")
features_train = extract_features(efficient_model, X_train)
features_test = extract_features(efficient_model, X_test)

print(f"📐 Features shape: {features_train.shape}")

# 3️⃣ Classifieur amélioré
classifier = Sequential([
    Dense(256, activation='relu', input_shape=(features_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

classifier.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Callbacks pour améliorer l'entraînement
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

print("🚀 Début de l'entraînement...")
history = classifier.fit(
    features_train, y_train, 
    epochs=30, 
    batch_size=16, 
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# 4️⃣ Évaluation améliorée
test_loss, test_accuracy = classifier.evaluate(features_test, y_test, verbose=0)
print(f"\n🎯 Accuracy sur le test set: {test_accuracy:.4f}")

y_pred_prob = classifier.predict(features_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_true, y_pred))

print("\n📈 Rapport de classification :")
print(classification_report(y_true, y_pred, target_names=classes))

# 5️⃣ Visualisation des résultats
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy during training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss during training')
plt.legend()

plt.tight_layout()
plt.show()