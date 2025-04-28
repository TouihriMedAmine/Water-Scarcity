import os
# import cgi # Plus nécessaire ici
import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from PIL import Image as PILImage # Plus nécessaire ici si l'image est lue via cv2 depuis le backend
# from http.server import BaseHTTPRequestHandler, HTTPServer # Supprimé
from tensorflow import keras

# Chemins relatifs depuis l'emplacement de app.py (supposé à la racine de WaterScarcity)
# Ajustez si nécessaire
cnn_model_path = os.path.join("Model_Water_Level", "Classification.keras")
unet_lake_model_path = os.path.join("Model_Water_Level", "Lake_prediction.keras")
unet_river_model_path = os.path.join("Model_Water_Level", "River_prediction.keras")
unet_harbor_model_path = os.path.join("Model_Water_Level", "harbor_prediction.keras")


# Vérifiez si les fichiers existent avant de charger
if not os.path.exists(cnn_model_path):
    raise FileNotFoundError(f"Modèle CNN non trouvé: {cnn_model_path}")
if not os.path.exists(unet_lake_model_path):
    raise FileNotFoundError(f"Modèle U-Net Lac non trouvé: {unet_lake_model_path}")
if not os.path.exists(unet_river_model_path):
    raise FileNotFoundError(f"Modèle U-Net Rivière non trouvé: {unet_river_model_path}")
if not os.path.exists(unet_harbor_model_path):
    raise FileNotFoundError(f"Modèle U-Net Port non trouvé: {unet_harbor_model_path}")


cnn_model = keras.models.load_model(cnn_model_path, compile=False)
unet_lake_model = keras.models.load_model(unet_lake_model_path, compile=False)
unet_river_model = keras.models.load_model(unet_river_model_path, compile=False)
unet_harbor_model = keras.models.load_model(unet_harbor_model_path, compile=False)


# Target image size for models
IMG_SIZE = (128, 128)

# Function to predict the class of the image using CNN
def predict_class(image):
    # ... existing code ...
    img_resized = cv2.resize(image, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    prediction = cnn_model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 0:
        return "harbor"
    elif predicted_class == 1:
        return "lake"
    else:
        return "river"

# Function to predict depth based on the class using U-Net
def predict_depth(image, predicted_class):
    # ... existing code ...
    if predicted_class == "lake":
        model = unet_lake_model
    elif predicted_class == "river":
        model = unet_river_model
    else: # harbor
        model = unet_harbor_model

    img_resized = cv2.resize(image, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    predicted_mask = model.predict(img_input)[0]

    # Convert to correct format for mask and ensure it's the right size
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    # Resize mask to original image size might be better if needed later,
    # but for blue intensity calculation, resizing to IMG_SIZE is consistent
    predicted_mask_resized = cv2.resize(predicted_mask, (IMG_SIZE[1], IMG_SIZE[0]))

    # Make sure blue_channel is also the right format and size
    # Use the already resized image (img_resized) for consistency
    blue_channel = (img_resized[:, :, 0] * 255).astype(np.uint8)

    # Now apply the mask
    water_region = cv2.bitwise_and(blue_channel, blue_channel, mask=predicted_mask_resized)

    # Calculate blue intensity
    non_zero_pixels = cv2.countNonZero(predicted_mask_resized)
    if non_zero_pixels > 0:
        # Use the masked region for sum calculation
        blue_intensity = np.sum(water_region[predicted_mask_resized > 0]) / non_zero_pixels / 255.0
    else:
        blue_intensity = 0

    if blue_intensity < 0.33:
        depth_label = "Profondeur élevée" # Traduit
    elif blue_intensity < 0.66:
        depth_label = "Profondeur moyenne" # Traduit
    else:
        depth_label = "Faible profondeur" # Traduit

    return depth_label, round(blue_intensity, 2)

# --- Supprimer la classe SimpleHTTPRequestHandler ---
# class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
#    ... (tout le code de la classe est supprimé) ...

# --- Supprimer la fonction run et le bloc main ---
# def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8888):
#    ... (tout le code de la fonction est supprimé) ...
#
# if __name__ == "__main__":
#    run()