# Étape 7 - IHM & Explicabilité (Streamlit + SHAP + Grad-CAM)
import io
import os
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import load_model

st.set_page_config(page_title='IHM - Explicabilité', layout='wide')

# --- Configuration utilisateur ---
MODEL_PATH = 'best_classifier.h5'
USE_GPU = False
BACKGROUND_SAMPLE_SIZE = 50
MC_DROPOUT_T = 20

# --- Sidebar : options ---
st.sidebar.header('Options')
use_gradcam = st.sidebar.checkbox('Activer Grad-CAM (localisation)', value=True)
use_shap = st.sidebar.checkbox('Activer SHAP (importance features)', value=True)
mc_t = st.sidebar.number_input('MC Dropout passes', value=MC_DROPOUT_T, min_value=5, max_value=200, step=5)
bg_size = st.sidebar.number_input('Background samples pour SHAP', value=BACKGROUND_SAMPLE_SIZE, min_value=10, max_value=200, step=10)

# --- Utilities ---
@st.cache_resource
def load_models():
    # efficient_model : extrait features depuis l'image (include_top=False)
    eff = EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')

    # classifier : modèle dense entraîné (prend en entrée features de dim eff.output)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH}. Place ton fichier best_classifier.h5 dans le dossier.")
        raise FileNotFoundError(MODEL_PATH)
    clf = load_model(MODEL_PATH)

    # construire un modèle complet image -> class_prob pour Grad-CAM
    input_img = eff.input
    feats = eff.output  # tensor shape (None, 1280)

    # wrapper : appliquer classifier sur feats
    outputs = clf(feats)
    full_model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return eff, clf, full_model

def preprocess_pil(img_pil, target_size=(224,224)):
    img = img_pil.convert('RGB').resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def get_features(eff_model, img_array):
    feats = eff_model.predict(img_array)
    return feats

def predict_with_uncertainty(clf_model, features, T=20):
    preds = np.stack([clf_model(features, training=True).numpy() for _ in range(T)], axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

def build_shap_explainer(clf_model, background_features):
    # S'assurer que les features sont en 2D
    if len(background_features.shape) > 2:
        background_features = background_features.reshape(background_features.shape[0], -1)
    
    def predict_fn(X):
        # Reshape si nécessaire
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        preds = clf_model.predict(X, verbose=0)
        return preds

    explainer = shap.KernelExplainer(predict_fn, background_features)
    return explainer

def compute_gradcam(full_model, img_array, class_index=1, last_conv_layer_name="top_conv"):
    possible_layer_names = ["top_conv", "block6e_expand_conv", "block6d_expand_conv"]
    
    for layer_name in possible_layer_names:
        for layer in full_model.layers:
            if layer_name in layer.name:
                last_conv_layer_name = layer.name
                break
    
    grad_model = tf.keras.models.Model(
        [full_model.inputs], [full_model.get_layer(last_conv_layer_name).output, full_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-9)
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap

def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5, cmap=cv2.COLORMAP_JET):
    img = np.array(img_pil.convert('RGB').resize((224,224)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

@st.cache_data(ttl=3600)
def prepare_background(eff_model, bg_size=50):
    data_dir = r"C:\Users\ACER\Desktop\new folder\Saoussen\mmehela\projet\chestxray_diagnostic_system\data\raw\images"

    imgs = []
    if os.path.exists(data_dir):
        for cls in os.listdir(data_dir):
            folder = os.path.join(data_dir, cls)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if len(imgs) >= bg_size:
                    break
                try:
                    p = os.path.join(folder, f)
                    img = Image.open(p).convert('RGB').resize((224,224))
                    arr = image.img_to_array(img)
                    imgs.append(arr)
                except Exception:
                    continue
            if len(imgs) >= bg_size:
                break

    if len(imgs) == 0:
        raise RuntimeError('Aucun fichier trouvé dans data/raw/images pour construire background.')

    imgs = np.array(imgs)
    imgs = preprocess_input(imgs)
    feats = eff_model.predict(imgs, verbose=0)
    return feats

# --- Interface principale ---
st.title('Module IHM — Explicabilité (Étape 7)')
st.markdown("Charge une radiographie, obtiens la prédiction, l'incertitude, et des explications (SHAP + Grad-CAM).")

# --- Charge les modèles ---
with st.spinner('Chargement des modèles...'):
    eff_model, clf_model, full_model = load_models()
st.success('Modèles chargés')

# --- Préparer background SHAP ---
background_feats = None
if use_shap:
    try:
        with st.spinner('Préparation du background pour SHAP (features)...'):
            background_feats = prepare_background(eff_model, bg_size)
        st.success('Background SHAP prêt')
    except Exception as e:
        st.error('Impossible de préparer background SHAP: ' + str(e))
        use_shap = False

# --- Upload image ---
uploaded = st.file_uploader('Charger une image (jpg, png, bmp)', type=['jpg', 'jpeg', 'png', 'bmp'])

# --- Traitement de l'image ---
if uploaded is not None:
    img_pil = Image.open(io.BytesIO(uploaded.read()))
    st.subheader('Image chargée')
    st.image(img_pil, width=300)

    # Prétraitement
    arr = preprocess_pil(img_pil)

    # Features
    with st.spinner('Extraction des features via EfficientNetV2S...'):
        feats = get_features(eff_model, arr)
    st.write('Features shape:', feats.shape)

    # Prédiction & incertitude
    with st.spinner('Prédiction & estimation d\'incertitude (MC Dropout)...'):
        mean_prob, std_prob = predict_with_uncertainty(clf_model, feats, T=mc_t)
    
    prob_normal = mean_prob[0,0]
    prob_pneumonia = mean_prob[0,1]
    st.metric('Probabilité Normal', f"{prob_normal:.3f}")
    st.metric('Probabilité Pneumonia', f"{prob_pneumonia:.3f}")
    st.write('Incertitude (std) pour Pneumonia :', f"{std_prob[0,1]:.4f}")

    # Label final
    predicted_label = 'pneumonia' if prob_pneumonia > prob_normal else 'normal'
    st.info(f'Prédiction finale : **{predicted_label.upper()}** (score {max(prob_normal, prob_pneumonia):.3f})')

    # Grad-CAM
    if use_gradcam:
        try:
            with st.spinner('Calcul Grad-CAM...'):
                heatmap = compute_gradcam(full_model, arr, class_index=1 if predicted_label=='pneumonia' else 0)
                overlay = overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5)
            st.subheader('Grad-CAM (localisation des régions influentes)')
            st.image(overlay, width=350)
        except Exception as e:
            st.error('Erreur Grad-CAM: ' + str(e))

    # SHAP - VERSION CORRECTE ET COMPLÈTE
    if use_shap and background_feats is not None:
        try:
            with st.spinner('Calcul SHAP (peut être lent)...'):
                bg = background_feats[:min(background_feats.shape[0], bg_size)]
                
                # Reshape pour s'assurer que c'est 2D
                if len(feats.shape) > 2:
                    feats_flat = feats.reshape(feats.shape[0], -1)
                else:
                    feats_flat = feats
                    
                if len(bg.shape) > 2:
                    bg_flat = bg.reshape(bg.shape[0], -1)
                else:
                    bg_flat = bg
                
                explainer = build_shap_explainer(clf_model, bg_flat)
                shap_vals = explainer.shap_values(feats_flat, nsamples=100)

            st.subheader('Explication SHAP (importance des features)')
            
            # CORRECTION COMPLÈTE - Extraction des valeurs SHAP
            st.write("🔍 Debug - Shape complète SHAP:", shap_vals.shape)
            
            if len(shap_vals.shape) == 3:
                # Shape (1, 1280, 2) - cas normal
                class_idx = 1 if predicted_label == 'pneumonia' else 0
                vals = shap_vals[0, :, class_idx]  # Premier sample, toutes features, classe prédite
            elif len(shap_vals.shape) == 2:
                # Shape (1280, 2) - cas alternatif
                class_idx = 1 if predicted_label == 'pneumonia' else 0
                vals = shap_vals[:, class_idx]
            else:
                st.error(f"Shape SHAP non supportée: {shap_vals.shape}")
                vals = np.array([])
            
            # AFFICHAGE DU GRAPHIQUE SHAP
            if len(vals) > 0:
                k = min(20, len(vals))
                top_idx = np.argsort(-np.abs(vals))[:k]
                
                plot_values = vals[top_idx][::-1]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Créer les couleurs
                colors = ['red' if x < 0 else 'green' for x in plot_values]
                
                ax.barh(range(k), plot_values, color=colors)
                ax.set_yticks(range(k))
                ax.set_yticklabels([f'Feature_{i}' for i in top_idx[::-1][:k]])
                ax.set_xlabel('SHAP Value (Impact sur la prédiction)')
                ax.set_title(f'Top {k} Features Influentes - Classe: {predicted_label.upper()}')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Ajouter des valeurs sur les barres
                for i, v in enumerate(plot_values):
                    ax.text(v, i, f'{v:.3f}', va='center', 
                           ha='left' if v < 0 else 'right',
                           fontsize=9, color='white' if abs(v) > 0.1 else 'black')
                
                st.pyplot(fig)
                
                st.success(f"✅ SHAP a analysé {len(vals)} features - {k} plus influentes affichées")
                
                # Interprétation des résultats
                st.subheader("📊 Interprétation SHAP")
                positive_impact = np.sum(vals > 0)
                negative_impact = np.sum(vals < 0)
                st.write(f"- **Features avec impact positif** : {positive_impact} (favorisent '{predicted_label}')")
                st.write(f"- **Features avec impact négatif** : {negative_impact} (défavorisent '{predicted_label}')")
                st.write(f"- **Features neutres** : {len(vals) - positive_impact - negative_impact}")
                
            else:
                st.error("❌ Aucune valeur SHAP disponible")

        except Exception as e:
            st.error('Erreur SHAP: ' + str(e))
            import traceback
            st.write("Détails de l'erreur:", traceback.format_exc())

# Footer
st.markdown('---')
st.caption('IHM réalisée pour l\'Étape 7 — Explicabilité & Interface.')