# ===========================================
# ÉTAPE 8 - Intégration & Gestion des connaissances
# ===========================================

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

# ============================================================
# 1) Chargement des modèles Vision + Décision (Étapes 4, 5, 6)
# ============================================================

MODEL_PATH = "best_classifier.h5"

def load_system_models():
    print("[INFO] Chargement du modèle EfficientNetV2S + classifieur...")
    eff = EfficientNetV2S(weights="imagenet", include_top=False, pooling="avg")

    clf = load_model(MODEL_PATH)
    print("[INFO] Modèles chargés avec succès !")

    return eff, clf

# ============================================================
# 2) Prétraitement : même pipeline que l’IHM (Étape 7)
# ============================================================

def preprocess_image(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr, img

# ============================================================
# 3) Inférence du modèle de décision
# ============================================================

def predict(eff, clf, img_array):
    feats = eff.predict(img_array)
    preds = clf.predict(feats)

    prob_normal = preds[0][0]
    prob_pneumonia = preds[0][1]

    label = "pneumonia" if prob_pneumonia > prob_normal else "normal"
    
    return label, prob_normal, prob_pneumonia

# ============================================================
# 4) MOTEUR DE CONNAISSANCES (règles médicales)
# ============================================================

def medical_recommendation(label, prob):
    """
    Module gestion de connaissances : règles médicales simples.
    """

    rules = {
        "normal": (
            "La radiographie semble normale.",
            "Aucune anomalie détectée par le système.",
            "Si symptômes persistants → contrôle clinique conseillé."
        ),
        "pneumonia": (
            "Suspicion de pneumonie détectée.",
            "Recommandation : orientation vers un spécialiste.",
            "Une confirmation via analyse clinique est fortement conseillée."
        )
    }

    recommendations = rules[label]

    return {
        "diagnostic": recommendations[0],
        "details": recommendations[1],
        "clinique": recommendations[2],
        "confiance": round(prob, 3)
    }

# ============================================================
# 5) Pipeline complet (end-to-end)
# ============================================================

def run_system(img_path):
    """
    Pipeline complet utilisé dans les tests finaux de l'Étape 8 :
    image → features → prédiction → recommandation médicale
    """

    print(f"[INFO] Analyse de l'image : {img_path}")

    eff, clf = load_system_models()

    arr, img_pil = preprocess_image(img_path)

    label, p_norm, p_pneu = predict(eff, clf, arr)

    prob = p_pneu if label == "pneumonia" else p_norm

    reco = medical_recommendation(label, prob)

    print("\n====================")
    print("🔍 Résultat final")
    print("====================")
    print("Classe prédite :", label.upper())
    print("Confiance :", prob)
    print("\n📌 Recommandation médicale :")
    print(reco["diagnostic"])
    print(reco["details"])
    print(reco["clinique"])

    return {
        "label": label,
        "prob": prob,
        "recommandation": reco
    }

# ============================================================
# 6) Tests automatiques (stabilité)
# ============================================================

def test_system():
    print("\n🧪 Test automatique du système (Étape 8)")

    try:
        run_system("test_image.jpg")
        print("TEST OK – Pipeline fonctionnel ✔")
    except Exception as e:
        print("TEST ÉCHOUÉ ❌", e)


# ============================================================
# Exécution directe
# ============================================================

if __name__ == "__main__":
    print("=== ETAPE 8 : Intégration & Moteur de connaissances ===")
    test_system()
