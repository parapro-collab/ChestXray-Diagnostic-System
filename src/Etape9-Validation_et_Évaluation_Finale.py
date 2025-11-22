"""

- Évalue le modèle sur test/
- Calcule metrics (accuracy, precision, recall, f1, auc)
- Génère matrice de confusion + ROC
- Sauvegarde CSV résumé
- Sauvegarde exemples mal classés + GradCAM
"""

import os
import numpy as np
import csv
import shutil
from glob import glob
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
MODEL_PATH = "best_classifier.h5"
TEST_DIR = "test"  # test/normal  test/pneumonia
OUTPUT_DIR = "step9_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MISCLASS_DIR = os.path.join(OUTPUT_DIR, "misclassified")
os.makedirs(MISCLASS_DIR, exist_ok=True)

# --------------------------------------------------------------
# 1) CONSTRUCTION DU MODELE COMBINE (features + GradCAM)
# --------------------------------------------------------------
def build_combined_model():
    """
    Création d’un modèle :
    image -> EfficientNetV2S (pooling=None) -> GAP -> classifieur
    Retourne : (prédictions, feature_maps)
    """

    print("[INFO] Construction modèle combiné...")

    # EfficientNetV2S pour GradCAM (pooling=None)
    base_spatial = EfficientNetV2S(
        weights="imagenet", include_top=False, pooling=None
    )
    base_spatial.trainable = False

    # Entrée image
    img_input = Input(shape=(224, 224, 3))

    # Feature maps 7x7x1280
    feature_maps = base_spatial(img_input)

    # GAP → vecteur 1280
    pooled = GlobalAveragePooling2D()(feature_maps)

    # Charger ton classifieur
    classifier = load_model(MODEL_PATH)

    # Prédictions finales
    preds = classifier(pooled)

    # Modèle final
    combined_model = Model(
        inputs=img_input,
        outputs=[preds, feature_maps],
        name="combined_model"
    )

    print("[INFO] Modèle combiné prêt.")
    return combined_model

# --------------------------------------------------------------
# 2) PREPROCESSING
# --------------------------------------------------------------
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr, img

# --------------------------------------------------------------
# 3) PREDICTIONS
# --------------------------------------------------------------
def predict_on_image_combined(model, img_path):
    arr_pre, _ = load_and_preprocess(img_path)
    preds, feat_maps = model.predict(arr_pre, verbose=0)
    preds = preds[0]         # [p_normal, p_pneumonia]
    feat_maps = feat_maps[0] # (H,W,C)
    return preds, feat_maps

# --------------------------------------------------------------
# 4) GRAD-CAM
# --------------------------------------------------------------
def make_gradcam_heatmap(model, img_path, target_class=1):
    arr_pre, _ = load_and_preprocess(img_path)
    img_tensor = tf.convert_to_tensor(arr_pre)

    with tf.GradientTape() as tape:
        preds, feature_maps = model(img_tensor)
        loss = preds[:, target_class]

    grads = tape.gradient(loss, feature_maps)

    pooled_grads = tf.reduce_mean(grads, axis=(1,2))[0]
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()

    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    heatmap = tf.image.resize(heatmap[..., np.newaxis], (224,224)).numpy()
    return heatmap.squeeze()

def save_gradcam_on_image(img_pil, heatmap, out_path, alpha=0.5):
    import matplotlib.cm as cm
    img = np.array(img_pil).astype(np.uint8)
    cmap = cm.jet(heatmap)[...,:3]
    cmap = (cmap * 255).astype(np.uint8)
    blended = (img * (1 - alpha) + cmap * alpha).astype(np.uint8)
    plt.imsave(out_path, blended)

# --------------------------------------------------------------
# 5) EVALUATION COMPLETE
# --------------------------------------------------------------
def evaluate_and_report(model, test_dir, output_dir):
    classes = ["normal", "pneumonia"]

    y_true, y_pred, y_scores = [], [], []
    misclassified = []

    for i, cls in enumerate(classes):
        folder = os.path.join(test_dir, cls)
        if not os.path.isdir(folder):
            print(f"[WARN] Dossier manquant : {folder}")
            continue

        for f in sorted(glob(os.path.join(folder, "*"))):
            try:
                preds, _ = predict_on_image_combined(model, f)
            except:
                continue

            true = i
            pred = int(np.argmax(preds))
            score_pneu = float(preds[1])

            y_true.append(true)
            y_pred.append(pred)
            y_scores.append(score_pneu)

            if true != pred:
                misclassified.append({
                    "path": f,
                    "true": classes[true],
                    "pred": classes[pred]
                })

    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    cm = confusion_matrix(y_true, y_pred)

    # Save CSV
    with open(os.path.join(output_dir, "evaluation_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["accuracy", acc])
        w.writerow(["precision", prec])
        w.writerow(["recall", rec])
        w.writerow(["f1", f1])
        w.writerow(["auc", auc])
        w.writerow(["confusion_matrix", cm.tolist()])

    # Save confusion matrix
    plot_confusion_matrix(cm, classes, os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc(y_true, y_scores, os.path.join(output_dir, "roc_curve.png"))

    # Save misclassified
    for idx, m in enumerate(misclassified):
        shutil.copy(m["path"], os.path.join(MISCLASS_DIR, f"mis_{idx}.jpg"))

        # GradCAM
        arr_pre, img_pil = load_and_preprocess(m["path"])
        heat = make_gradcam_heatmap(model, m["path"], target_class=1)
        save_gradcam_on_image(img_pil, heat,
                              os.path.join(MISCLASS_DIR, f"mis_{idx}_gradcam.png"))

    print("\n=== ÉVALUATION FINALE ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)
    print("Confusion matrix:\n", cm)
    print(f"{len(misclassified)} images mal classées.")
    print("Résultats enregistrés dans :", output_dir)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion": cm,
        "misclassified": misclassified
    }

# --------------------------------------------------------------
# 6) PLOTS
# --------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_scores, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    combined_model = build_combined_model()
    results = evaluate_and_report(combined_model, TEST_DIR, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Step 9 Final Evaluation\n")
        f.write(str(results))

    print("[DONE] Étape 9 terminée.")
