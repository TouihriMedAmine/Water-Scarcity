import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import json

# Paramètres globaux communs
IMG_HEIGHT = 128
IMG_WIDTH = 256
LON_MIN, LON_MAX = -125, -66.5
LAT_MIN, LAT_MAX = 24, 50
IMG_MIN_NORM = 0.0  # Renommé pour éviter confusion avec MIN_PHYSICAL_VALUE
IMG_MAX_NORM = 1.0  # Renommé pour éviter confusion avec MAX_PHYSICAL_VALUE

# Configuration spécifique à chaque modèle
MODEL_CONFIGS = {
    "AvgSurfT": {
        "MODEL_SUBDIR": "AvgSurfT",
        "DATA_DIR_SUFFIX": "AvgSurfT",
        "MODEL_FILENAME": "modele_unet_AvgSurfT.h5",
        "MIN_PHYSICAL_VALUE": 260,
        "MAX_PHYSICAL_VALUE": 320,
        "UNIT": "K",
        "FILENAME_PREFIX": "AvgSurfT",
        "PHYSICAL_PREDICTION_FORMULA": lambda pred_norm, min_val, max_val: max_val + 23.3 - pred_norm * (max_val - min_val)
    },
    "PotEvap": {
        "MODEL_SUBDIR": "PotEvap",
        "DATA_DIR_SUFFIX": "PotEvap",
        "MODEL_FILENAME": "modele_unet_PotEvap.h5",
        "MIN_PHYSICAL_VALUE": 0,
        "MAX_PHYSICAL_VALUE": 300,
        "UNIT": "W m-2",
        "FILENAME_PREFIX": "PotEvap",
        "PHYSICAL_PREDICTION_FORMULA": lambda pred_norm, min_val, max_val: pred_norm * max_val + 6.02
    },
    "Rainf": {
        "MODEL_SUBDIR": "Rainf",
        "DATA_DIR_SUFFIX": "Rainf",
        "MODEL_FILENAME": "modele_unet_Rainf.h5",
        "MIN_PHYSICAL_VALUE": 0,
        "MAX_PHYSICAL_VALUE": 800,
        "UNIT": "mm",
        "FILENAME_PREFIX": "Rainf",
        "PHYSICAL_PREDICTION_FORMULA": lambda pred_norm, min_val, max_val: max_val - pred_norm * max_val
    },
    "SoilM": {
        "MODEL_SUBDIR": "SoilM",
        "DATA_DIR_SUFFIX": "SoilM_0_100cm",
        "MODEL_FILENAME": "modele_unet_SoilM_0_100cm.h5",
        "MIN_PHYSICAL_VALUE": 0,
        "MAX_PHYSICAL_VALUE": 500,
        "UNIT": "kg m-2",
        "FILENAME_PREFIX": "SoilM_0_100cm",
        "PHYSICAL_PREDICTION_FORMULA": lambda pred_norm, min_val, max_val: max_val - 52 - pred_norm * max_val
    }
}

# Chemins de base
# Le script Unified_Server.py est supposé être dans Model_example/
CURRENT_SCRIPT_DIR = os.path.dirname(__file__)
# Les données sont dans Water-Scarcity/visualization_outputUp_2019_2024/
DATA_DIR_BASE = os.path.join(CURRENT_SCRIPT_DIR, "visualization_outputUp_2019_2024")
# Les modèles .h5 sont dans Model_example/ModelType/modele_unet_ModelType.h5
MODEL_FILES_BASE_DIR = CURRENT_SCRIPT_DIR

# Cache pour les modèles chargés
LOADED_MODELS_CACHE = {}

# Fonctions utilitaires communes

def get_model(model_name, config):
    """Charge un modèle TensorFlow ou le récupère du cache."""
    if model_name not in LOADED_MODELS_CACHE:
        model_path = os.path.join(MODEL_FILES_BASE_DIR, config["MODEL_SUBDIR"], config["MODEL_FILENAME"])
        try:
            print(f"Chargement du modèle pour {model_name} depuis {model_path}...")
            LOADED_MODELS_CACHE[model_name] = load_model(model_path, compile=False)
            print(f"Modèle {model_name} chargé.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name} depuis {model_path}: {e}")
            return None
    return LOADED_MODELS_CACHE[model_name]

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
    normalized_year = (year - 2019) / 5  # Adapter si la plage d'années change
    norm_month = (month - 1) / 11
    norm_day = (day - 1) / 30
    norm_doy = (day_of_year - 1) / 364.25 # Utiliser 364.25 pour la moyenne sur plusieurs années
    return np.array([
        normalized_year, norm_month, norm_day, norm_doy,
        month_sin, month_cos, day_sin, day_cos
    ], dtype=np.float32)

def find_jmoins1_image_path(date_cible, model_data_dir, filename_prefix):
    """Trouve le chemin de l'image J-1 pour la date cible et le modèle spécifié."""
    date_jmoins1 = date_cible - timedelta(days=1)
    img_filename = f"{filename_prefix}_{date_jmoins1.strftime('%Y%m%d')}.png"
    path = os.path.join(model_data_dir, img_filename)
    if os.path.exists(path):
        return path
    print(f"Image J-1 non trouvée : {path}")
    return None

def load_and_preprocess_image(img_path):
    """Charge et normalise une image PNG (0-1)."""
    try:
        img = plt.imread(img_path)
        if len(img.shape) > 2 and img.shape[2] > 1: # Convertir en niveaux de gris si RGBA ou RGB
            img = np.mean(img, axis=2)
        
        # S'assurer que l'image est bien en 2D avant d'ajouter l'axe des canaux
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        elif len(img.shape) == 3 and img.shape[2] == 1:
            pass # Déjà dans le bon format
        else:
            raise ValueError(f"Format d'image inattendu après lecture/conversion: {img.shape}")

        img_resized = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]).numpy()
        # Normalisation basée sur les constantes globales
        img_norm = (img_resized - IMG_MIN_NORM) / (IMG_MAX_NORM - IMG_MIN_NORM)
        img_norm = np.clip(img_norm, 0, 1)
        return img_norm
    except Exception as e:
        print(f"Erreur lors du chargement ou du prétraitement de l'image {img_path}: {e}")
        return None

def coords_to_indices(lon, lat):
    """Convertit les coordonnées lon/lat en indices x/y de l'image."""
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (IMG_WIDTH - 1))
    y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (IMG_HEIGHT - 1)) # Inversion de y car l'origine est en haut à gauche
    x = max(0, min(x, IMG_WIDTH - 1))
    y = max(0, min(y, IMG_HEIGHT - 1))
    return x, y

def get_all_predictions(date_str: str, lon_str: str, lat_str: str) -> dict:
    """
    Calcule les prédictions pour tous les modèles configurés pour une date,
    une longitude et une latitude données.

    Args:
        date_str (str): La date cible au format "YYYY-MM-DD".
        lon_str (str): La longitude sous forme de chaîne de caractères.
        lat_str (str): La latitude sous forme de chaîne de caractères.

    Returns:
        dict: Un dictionnaire contenant les prédictions pour chaque modèle,
              ou un message d'erreur si les paramètres sont invalides.
    """
    if not all([date_str, lon_str, lat_str]):
        return {"error": "Paramètres 'date_str', 'lon_str', et 'lat_str' requis."}

    try:
        date_cible = datetime.strptime(date_str, "%Y-%m-%d")
        lon = float(lon_str)
        lat = float(lat_str)
    except ValueError:
        return {"error": "Format de date invalide (YYYY-MM-DD) ou coordonnées invalides."}

    all_predictions_result = {}
    date_features_np = date_to_features(date_cible)[np.newaxis, ...] # Ajouter dimension batch

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\nTraitement du modèle : {model_name}")
        
        model_instance = get_model(model_name, config)
        if model_instance is None:
            all_predictions_result[model_name] = {"error": "Modèle non chargé.", "unit": config["UNIT"]}
            continue

        model_specific_data_dir = os.path.join(DATA_DIR_BASE, config["DATA_DIR_SUFFIX"])
        img_path_jmoins1 = find_jmoins1_image_path(date_cible, model_specific_data_dir, config["FILENAME_PREFIX"])

        if img_path_jmoins1 is None:
            all_predictions_result[model_name] = {"error": "Image J-1 non trouvée.", "unit": config["UNIT"]}
            continue
        
        img_jmoins1_processed = load_and_preprocess_image(img_path_jmoins1)
        if img_jmoins1_processed is None:
            all_predictions_result[model_name] = {"error": "Erreur de traitement de l'image J-1.", "unit": config["UNIT"]}
            continue
        
        img_jmoins1_batch = img_jmoins1_processed[np.newaxis, ...] # Ajouter dimension batch

        try:
            # La prédiction attend une liste d'entrées si le modèle a plusieurs têtes d'entrée
            pred_img_normalized = model_instance.predict([img_jmoins1_batch, date_features_np], verbose=0) # Ajout de verbose=0 pour moins de logs TF
        except Exception as e:
            print(f"Erreur lors de la prédiction pour {model_name}: {e}")
            all_predictions_result[model_name] = {"error": f"Erreur de prédiction: {str(e)}", "unit": config["UNIT"]}
            continue
            
        idx_x, idx_y = coords_to_indices(lon, lat)
        
        # S'assurer que pred_img_normalized a la bonne forme avant l'indexation
        # typiquement (batch, height, width, channels)
        if pred_img_normalized.ndim == 4 and pred_img_normalized.shape[0] == 1:
            pred_norm_value = float(pred_img_normalized[0, idx_y, idx_x, 0])
        else:
            print(f"Shape de prédiction inattendue pour {model_name}: {pred_img_normalized.shape}")
            all_predictions_result[model_name] = {"error": "Shape de prédiction inattendue.", "unit": config["UNIT"]}
            continue

        raw_physical_prediction = config["PHYSICAL_PREDICTION_FORMULA"](
            pred_norm_value, 
            config["MIN_PHYSICAL_VALUE"], 
            config["MAX_PHYSICAL_VALUE"]
        )
        physical_prediction = round(raw_physical_prediction, 3)
        
        all_predictions_result[model_name] = {
            "prediction": physical_prediction,
            "unit": config["UNIT"]
        }
        print(f"Prédiction pour {model_name} ({lat}, {lon}) à {date_str}: {physical_prediction} {config['UNIT']}")

    return all_predictions_result
