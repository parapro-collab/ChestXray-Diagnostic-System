#   Prétraitement du dataset ChestX-ray14 (NIH)


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Chargement du fichier d’annotations


data_path = "data/"
csv_file = os.path.join(data_path, "Data_Entry_2017.csv")

data = pd.read_csv(csv_file)
print("Nombre total d'images :", len(data))
print("\nExemple :\n", data.head())


#  Nettoyage et formatage des labels


# Remplacement du label 'No Finding' par 'Normal'
data['Finding Labels'] = data['Finding Labels'].replace('No Finding', '')
data['Finding Labels'] = data['Finding Labels'].apply(lambda x: x.split('|') if x != '' else ['Normal'])

# Encodage multi-label


mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['Finding Labels'])

print("\nListe des classes :", list(mlb.classes_))


# Séparation train/test par patient


patients = data['Patient ID'].unique()
train_pat, test_pat = train_test_split(patients, test_size=0.2, random_state=42)

train_data = data[data['Patient ID'].isin(train_pat)]
test_data  = data[data['Patient ID'].isin(test_pat)]

print(f"\nTrain : {len(train_data)} images")
print(f"Test  : {len(test_data)} images")

# Fonction de prétraitement d’une image


def preprocess_image(img_path, img_size=(224, 224)):
    """
    Lecture, conversion en N&B, égalisation d’histogramme et normalisation
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


# Générateurs d’images pour l’entraînement


train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

# Sauvegarde des fichiers préparés


train_data.to_csv(os.path.join(data_path, "train_clean.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test_clean.csv"), index=False)

print("\n✅ Prétraitement terminé avec succès !")
