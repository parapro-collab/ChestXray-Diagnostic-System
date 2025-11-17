import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score

# ----------------------------------------------------------
# 1️⃣ Chargement du dataset
# ----------------------------------------------------------

dataset_dir = r"C:\Users\ACER\Desktop\new folder\Saoussen\mmehela\projet\chestxray_diagnostic_system\data\raw\images"
classes = ["normal", "pneumonia"]

X, y = [], []

for idx, label in enumerate(classes):
    folder = os.path.join(dataset_dir, label)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Le dossier n'existe pas : {folder}")

    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)

        # Charger et préparer l'image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        X.append(img_array)
        y.append(idx)

X = np.array(X)
y = np.array(y)

# Normalisation adaptée EfficientNet
X = preprocess_input(X)

# One-hot encoding
y_cat = to_categorical(y, num_classes=2)

# Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

print("✔️ Dataset chargé :")
print(" - Taille totale :", len(X))
print(" - Train :", len(X_train))
print(" - Test :", len(X_test))

# ----------------------------------------------------------
# 2️⃣ Extraction des features avec EfficientNetV2S
# ----------------------------------------------------------

efficient_model = EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
print("✔️ EfficientNetV2S chargé")

# Extraction des caractéristiques
features_train = efficient_model.predict(X_train)
features_test = efficient_model.predict(X_test)

print("✔️ Features extraites :")
print(" - Shape train :", features_train.shape)
print(" - Shape test :", features_test.shape)

# ----------------------------------------------------------
# 3️⃣ Classifieur Dense
# ----------------------------------------------------------

classifier = Sequential([
    Dense(256, activation='relu', input_shape=(features_train.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔧 Entraînement du modèle...\n")

history = classifier.fit(
    features_train, y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# ----------------------------------------------------------
# 4️⃣ Évaluation + métriques médicales
# ----------------------------------------------------------

y_pred_prob = classifier.predict(features_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📌 MATRICE DE CONFUSION")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Sensibilité = Recall class "pneumonia"
sensibilite = recall_score(y_true, y_pred, pos_label=1)

# Spécificité = recall class "normal"
specificite = recall_score(y_true, y_pred, pos_label=0)

print(f"\n📌 Sensibilité (Recall Pneumonia) : {sensibilite:.4f}")
print(f"📌 Spécificité (Recall Normal)    : {specificite:.4f}")

print("\n📌 RAPPORT DE CLASSIFICATION")
print(classification_report(y_true, y_pred, target_names=classes))

print("\n🎉 Étape 6 terminée avec succès !")
