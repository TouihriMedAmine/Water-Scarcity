from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Paramètres de la grille et normalisation
IMG_HEIGHT = 128
IMG_WIDTH = 256
LON_MIN, LON_MAX = -125, -66.5
LAT_MIN, LAT_MAX = 24, 50
DATA_DIR = os.path.join(os.path.dirname(__file__), "../visualization_outputUp_2019_2024/PotEvap")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modele_unet_PotEvap.h5")
IMG_MIN = 0.0  # À adapter si tu as utilisé d'autres bornes
IMG_MAX = 1.0  # À adapter si tu as utilisé d'autres bornes

# Chargement du modèle une seule fois
model = load_model(MODEL_PATH, compile=False)

app = Flask(__name__)

def extract_date_from_filename(filename):
    """Extrait la date du nom de fichier (ex: PotEvap_20190101.png)"""
    basename = os.path.basename(filename)
    try:
        date_str = basename.split("_")[1].split(".")[0]
        return datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        return None

def date_to_features(date_obj):
    """Convertit une date en features numériques cycliques et normalisées."""
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    day_of_year = date_obj.timetuple().tm_yday
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    normalized_year = (year - 2019) / 5  # Adapter selon tes données
    norm_month = (month - 1) / 11
    norm_day = (day - 1) / 30
    norm_doy = (day_of_year - 1) / 364.25
    return np.array([
        normalized_year, norm_month, norm_day, norm_doy,
        month_sin, month_cos, day_sin, day_cos
    ], dtype=np.float32)

def find_jmoins1_image_path(date_cible):
    """Trouve le chemin de l'image J-1 pour la date cible."""
    date_jmoins1 = date_cible - timedelta(days=1)
    filename = f"PotEvap_{date_jmoins1.strftime('%Y%m%d')}.png"
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    return None

def load_and_preprocess_image(img_path):
    """Charge et normalise une image PNG (0-1)."""
    img = plt.imread(img_path)
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = np.mean(img, axis=2)
    img_resized = tf.image.resize(img[:, :, np.newaxis], [IMG_HEIGHT, IMG_WIDTH]).numpy()
    img_norm = (img_resized - IMG_MIN) / (IMG_MAX - IMG_MIN)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm

def coords_to_indices(lon, lat):
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (IMG_WIDTH - 1))
    y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (IMG_HEIGHT - 1))
    x = max(0, min(x, IMG_WIDTH - 1))
    y = max(0, min(y, IMG_HEIGHT - 1))
    return x, y

@app.route('/predict', methods=['GET'])
def predict():
    date_str = request.args.get('date')
    lon = float(request.args.get('lon'))
    lat = float(request.args.get('lat'))

    try:
        date_cible = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return jsonify({"error": "Format de date invalide. Utilise YYYY-MM-DD."}), 400

    img_path = find_jmoins1_image_path(date_cible)
    if img_path is None:
        return jsonify({"error": "Image J-1 non trouvée."}), 404

    img_jmoins1 = load_and_preprocess_image(img_path)[np.newaxis, ...]
    date_features = date_to_features(date_cible)[np.newaxis, ...]
    pred_img = model.predict([img_jmoins1, date_features])
    x, y = coords_to_indices(lon, lat)
    pred_norm = float(pred_img[0, y, x, 0])
    return jsonify({"prediction": pred_norm})

if __name__ == '__main__':
    app.run(debug=True)