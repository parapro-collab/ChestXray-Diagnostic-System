# src/feature_extraction.py
import os
# src/feature_extraction.py
import os
import glob
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import json
import pickle
from pathlib import Path

class ChestXRayFeatureExtractor:
    def __init__(self):
        print("🔄 Initialisation de l'extracteur de features...")
        
        # Modèle ResNet50 pré-entraîné
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Transformations pour images médicales
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_deep_features(self, image):
        """Extraire features Deep Learning depuis ResNet50"""
        try:
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(image_tensor)
            return features.squeeze().numpy()
        except Exception as e:
            print(f"❌ Erreur extraction deep features: {e}")
            return np.zeros(2048)
    
    def extract_handcrafted_features(self, image):
        """Extraire features traditionnelles pour radiographies"""
        try:
            # Histogrammes des canaux
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Texture et contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
            
            # Statistiques simples
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Combiner tous les features
            handcrafted = np.concatenate([
                hist_r.flatten()[:128],
                hist_g.flatten()[:128],
                hist_b.flatten()[:128],
                [edge_density, mean_intensity, std_intensity]
            ])
            
            return handcrafted
        except Exception as e:
            print(f"❌ Erreur extraction handcrafted features: {e}")
            return np.zeros(387)

def load_images_from_raw():
    """Charge les 100 images depuis le dossier raw/images"""
    
    raw_data_path = r"C:\Users\ACER\Desktop\new folder\Saoussen\mmehela\projet\chestxray_diagnostic_system\data\raw\images"
    
    print(f"📁 Chargement depuis: {raw_data_path}")
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Le chemin n'existe pas: {raw_data_path}")
        return [], [], []
    
    # Chercher toutes les images
    image_extensions = ['*.jpeg', '*.jpg', '*.png']
    all_images_paths = []
    
    for ext in image_extensions:
        images = glob.glob(os.path.join(raw_data_path, ext))
        all_images_paths.extend(images)
    
    # Prendre maximum 100 images
    images_to_process = all_images_paths[:100]
    print(f"✅ {len(images_to_process)} images trouvées")
    
    # Charger les images et déterminer les labels
    images_data = []
    labels = []
    valid_paths = []
    
    for img_path in images_to_process:
        try:
            image = cv2.imread(img_path)
            if image is not None:
                images_data.append(image)
                valid_paths.append(img_path)
                
                # Déterminer le label depuis le nom du fichier
                filename = os.path.basename(img_path).upper()
                if "NORM" in filename:
                    labels.append("NORMAL")
                elif "PNEUM" in filename or "VIRUS" in filename or "BACT" in filename:
                    labels.append("PNEUMONIA")
                else:
                    labels.append("UNKNOWN")
                    
        except Exception as e:
            print(f"❌ Erreur chargement {img_path}: {e}")
            continue
    
    return images_data, labels, valid_paths

def main():
    print("🎯 EXTRACTION FEATURES - 100 IMAGES")
    print("=" * 50)
    
    # 1. Charger les images
    print("📁 Étape 1: Chargement des images...")
    images, labels, image_paths = load_images_from_raw()
    
    if len(images) == 0:
        print("❌ Aucune image n'a pu être chargée!")
        return
    
    print(f"✅ {len(images)} images chargées avec succès")
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"📊 Répartition: {label_counts}")
    
    # 2. Initialiser l'extracteur
    print("🔧 Étape 2: Initialisation de l'extracteur...")
    extractor = ChestXRayFeatureExtractor()
    
    # 3. Extraction des features
    print("⚡ Étape 3: Extraction des features...")
    all_features = []
    metadata = []
    
    for i, (image, label, img_path) in enumerate(zip(images, labels, image_paths)):
        print(f"🔍 {i+1}/{len(images)}: {os.path.basename(img_path)}")
        
        try:
            # Extraction features Deep Learning
            deep_features = extractor.extract_deep_features(image)
            
            # Extraction features traditionnels
            handcrafted_features = extractor.extract_handcrafted_features(image)
            
            # Combiner les deux types de features
            combined_features = np.concatenate([deep_features, handcrafted_features])
            
            all_features.append(combined_features)
            metadata.append({
                "image_id": os.path.basename(img_path),
                "image_path": img_path,
                "label": label,
                "image_shape": image.shape,
                "features_dim": combined_features.shape[0]
            })
            
        except Exception as e:
            print(f"❌ Erreur extraction image {i+1}: {e}")
            continue
    
    if len(all_features) == 0:
        print("❌ Aucun feature n'a pu être extrait!")
        return
    
    # 4. Conversion en array numpy
    features_array = np.array(all_features)
    print(f"✅ Features extraits: {features_array.shape}")
    
    # 5. Réduction de dimensionnalité
    print("📉 Étape 4: Réduction de dimensionnalité...")
    
    # PCA - réduire à 50 dimensions maximum
    n_components = min(50, len(features_array) - 1, features_array.shape[1] - 1)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_array)
    
    # t-SNE pour visualisation
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features_array) - 1))
    features_tsne = tsne.fit_transform(features_pca)
    
    print(f"📊 Après PCA: {features_pca.shape}")
    print(f"📊 Après t-SNE: {features_tsne.shape}")
    
    # 6. Métriques de qualité
    print("📈 Étape 5: Calcul des métriques de qualité...")
    
    # Convertir labels en numérique
    label_map = {"NORMAL": 0, "PNEUMONIA": 1, "UNKNOWN": 2}
    numeric_labels = [label_map[label] for label in labels[:len(features_pca)]]
    
    # Silhouette score seulement si au moins 2 classes
    unique_labels = set(numeric_labels)
    if len(unique_labels) > 1:
        silhouette = silhouette_score(features_pca, numeric_labels)
    else:
        silhouette = 0.0
    
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    # 7. Sauvegarde des résultats
    print("💾 Étape 6: Sauvegarde des résultats...")
    
    output_dir = Path(r"C:\Users\ACER\Desktop\new folder\Saoussen\mmehela\projet\chestxray_diagnostic_system\features")
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder les vecteurs
    np.save(output_dir / "features_original.npy", features_array)
    np.save(output_dir / "features_pca.npy", features_pca)
    np.save(output_dir / "features_tsne.npy", features_tsne)
    
    # Sauvegarder métadonnées
    with open(output_dir / "metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder métriques
    metrics = {
        "n_images_processed": len(images),
        "n_features_original": int(features_array.shape[1]),
        "n_features_pca": int(features_pca.shape[1]),
        "silhouette_score": float(silhouette),
        "variance_explained": float(variance_explained),
        "labels_distribution": {str(k): int(v) for k, v in label_counts.items()}
    }
    
    with open(output_dir / "quality_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Sauvegarder modèle PCA
    with open(output_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    
    # 8. Rapport final
    print("\n" + "=" * 50)
    print("🎉 EXTRACTION TERMINÉE AVEC SUCCÈS!")
    print("=" * 50)
    print(f"📁 Résultats dans: {output_dir}")
    print(f"📊 Images traitées: {len(images)}")
    print(f"🔢 Dimensions: {features_array.shape[1]} → {features_pca.shape[1]}")
    print(f"📏 Silhouette: {silhouette:.3f}")
    print(f"📈 Variance: {variance_explained:.1%}")
    print(f"🏷️  Labels: {label_counts}")
    
    print(f"\n✅ LIVRABLES CRÉÉS:")
    print(f"  📄 features_original.npy - Extraction caractéristiques")
    print(f"  📄 features_pca.npy - Réduction dimensionnalité (PCA)")
    print(f"  📄 features_tsne.npy - Réduction dimensionnalité (t-SNE)")
    print(f"  📄 metadata.json - Représentations vectorielles")
    print(f"  📄 quality_metrics.json - Métriques de qualité")

if __name__ == "__main__":
    main()