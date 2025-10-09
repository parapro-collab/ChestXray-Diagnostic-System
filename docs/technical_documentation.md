Documentation technique
Objectif du projet
Ce projet vise à concevoir et développer un système intelligent d’aide au diagnostic radiologique, capable d’analyser automatiquement les radiographies thoraciques afin de détecter diverses pathologies pulmonaires.
Le système repose sur des modèles avancés de vision artificielle et intègre un module d’explicabilité basé sur GradCAM++, permettant aux médecins radiologues de comprendre et d’interpréter les décisions prises par le modèle.
Description du dataset
Le dataset ChestX-ray14 (NIH) comprend 112 120 radiographies thoraciques provenant de 30 805 patients.
Chaque image est annotée avec jusqu’à 14 pathologies, telles que : pneumonie, atélectasie, nodule pulmonaire, etc.
Les images sont au format PNG, et sont accompagnées d’un fichier CSV détaillant les labels et les métadonnées cliniques (Patient ID, pathologies associées, etc.).
Architecture du système
Le système se compose de trois modules principaux :
1.	Module Vision Artificielle
o	Prétraitement des images (redimensionnement, normalisation, égalisation d’histogramme)
o	Extraction de caractéristiques
o	Modèle de classification (EfficientNetV2, CNN ou Transformer comme ViT)
2.	Module d’Aide à la Décision
o	Interprétation des prédictions
o	Validation et contextualisation des résultats pour la pratique clinique
3.	Module Interface Homme-Machine
o	Visualisation interactive des résultats
o	Explicabilité des prédictions grâce à GradCAM++ (cartes thermiques illustrant les zones d’attention du modèle)
Schéma simplifié du flux :
Image radiographique → Prétraitement → Modèle IA → Prédiction → GradCAM++ → Interface utilisateur
Pipeline de prétraitement
1.	Chargement du CSV Data_Entry_2017.csv
2.	Nettoyage des labels et remplacement des valeurs manquantes
3.	Encodage multi-label (MultiLabelBinarizer)
4.	Séparation train/test par patient
5.	Prétraitement image :
o	Conversion en niveaux de gris
o	Égalisation d’histogramme
o	Redimensionnement en 224×224
o	Normalisation entre [0,1]
6.	Sauvegarde dans train_clean.csv et test_clean.csv

Explication  du code de prétraitement du dataset : ChestX-ray14(NIH)

1.	Importation des bibliothèques
Explication :
•	os : permet de manipuler les chemins et fichiers dans le système (pour lire et sauvegarder les données).
•	cv2 (OpenCV) : utilisé pour lire, convertir, redimensionner et traiter les images (ex. égalisation d’histogramme).
•	numpy (np) : pour les opérations mathématiques et les tableaux de pixels.
•	pandas (pd) : pour manipuler le fichier CSV (Data_Entry_2017.csv) contenant les métadonnées et labels.
•	train_test_split : pour diviser les patients en ensembles d’entraînement et de test.
•	MultiLabelBinarizer : encode les labels multi-pathologies (par ex. “Pneumonia|Effusion”) en vecteurs binaires.
•	ImageDataGenerator : permet d’appliquer des transformations d’augmentation de données sur les images (rotation, zoom…).
2.	Chargement du fichier d’annotations 
Explication :
•	On définit le chemin du dossier contenant les données (data/).
•	On construit le chemin complet du fichier CSV (Data_Entry_2017.csv).
•	pd.read_csv() lit ce fichier dans un DataFrame Pandas.
•	Ce fichier contient les colonnes :
o	Image Index : nom du fichier image (00000001_000.png)
o	Finding Labels : pathologies détectées (ex. “Infiltration|Effusion”)
o	Patient ID : identifiant unique du patient
•	On affiche le nombre total d’images et les premières lignes du fichier pour vérification.
3.	Nettoyage et formatage des labels
Explication :
•	Certaines images sont étiquetées "No Finding" → cela signifie qu’aucune pathologie n’a été détectée.
👉 On les remplace par le label 'Normal'.
•	Les autres images ont parfois plusieurs maladies séparées par le symbole | (multi-label).
👉 On transforme la chaîne "Pneumonia|Effusion" en une liste Python ['Pneumonia', 'Effusion'].
•	Si une cellule est vide (''), on assigne la classe ['Normal'].

4.	Encodage multi-label
Explication :
•	Le dataset est multi-label : une même image peut présenter plusieurs maladies.
•	MultiLabelBinarizer transforme la liste de labels de chaque image en un vecteur binaire.
Chaque position du vecteur correspond à une maladie dans mlb.classes_.

1.	Séparation train/test par patient
Explication :
•	On veut éviter que les images du même patient se retrouvent à la fois dans le train et dans le test (cela fausserait l’évaluation).
•	unique() récupère la liste de tous les patients.
•	train_test_split divise les patients en 80% entraînement et 20% test.
•	On sélectionne ensuite les images correspondant à chaque groupe via isin().
2.	Fonction de prétraitement d’une image
Étapes expliquées :
1.	Lecture de l’image :
cv2.imread() charge l’image depuis son chemin en niveaux de gris.
2.	Vérification :
Si le fichier est manquant ou corrompu → la fonction renvoie None.
3.	Égalisation d’histogramme :
cv2.equalizeHist() améliore le contraste, utile car les radios sont parfois trop sombres.
4.	Redimensionnement :
L’image est redimensionnée à (224, 224) pour correspondre à la taille d’entrée d’EfficientNetV2.
5.	Normalisation :
Division par 255.0 → les pixels passent de [0,255] à [0,1], ce qui accélère l’apprentissage.
6.	Ajout d’une dimension :
expand_dims transforme (224,224) en (224,224,1) (format attendu par le réseau CNN).

7.	Générateurs d’images pour l’entraînement
Explication :
•	ImageDataGenerator sert à augmenter artificiellement le nombre d’images d’entraînement :
o	rotation_range=10 : rotation aléatoire jusqu’à 10°
o	width_shift_range=0.1 / height_shift_range=0.1 : décalage horizontal et vertical
o	zoom_range=0.1 : zoom aléatoire
o	horizontal_flip=True : symétrie horizontale (comme si l’on regardait la radio de l’autre côté)
o	fill_mode='nearest' : complète les bords après transformation
•	test_datagen ne fait aucune transformation (les images de test doivent rester identiques pour l’évaluation).
8.	Sauvegarde des fichiers nettoyés
Explication :
•	On sauvegarde les fichiers CSV nettoyés et séparés (train_clean.csv et test_clean.csv).
•	Ces fichiers contiennent :
o	Le nom des images
o	Les labels formatés (listes)
o	Les ID patients
•	Ces fichiers serviront plus tard pour l’entraînement du modèle EfficientNetV2.
