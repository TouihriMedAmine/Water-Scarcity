import numpy as np
import cv2
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os # Import the os module

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the models relative to this script's directory
# Ensure these filenames match exactly what's in your Models_KERAS folder
lake_model_path = os.path.join(BASE_DIR, 'Lake_prediction.keras')
river_model_path = os.path.join(BASE_DIR, 'River_prediction.keras')
harbor_model_path = os.path.join(BASE_DIR, 'harbor_prediction.keras')
classifier_model_path = os.path.join(BASE_DIR, 'Classification.keras')


# ---- Load all models once at Django server startup ----
# Use the constructed paths to load the models
lake_model = tf.keras.models.load_model(lake_model_path)
river_model = tf.keras.models.load_model(river_model_path)
harbor_model = tf.keras.models.load_model(harbor_model_path)
classifier_model = tf.keras.models.load_model(classifier_model_path)

# Image size
img_size = (128, 128)

# --- 1. Classify uploaded image ---
def classify_uploaded_image(uploaded_file):
    # Read image from Django uploaded file
    img = Image.open(uploaded_file)
    img = img.convert('RGB')  # Ensure it's 3 channels
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = classifier_model.predict(img_array)
    predicted_class = np.argmax(prediction)

    class_mapping = {0: 'lake', 1: 'harbor', 2: 'river'}
    return class_mapping[predicted_class]

# --- 2. Prepare uploaded image for U-Net ---
def prepare_uploaded_image(uploaded_file):
    # Read image from Django uploaded file
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)

    # Generate mask
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    mask = mask / 255.0

    # Apply mask to blue channel
    water_region = cv2.bitwise_and(img_array, img_array, mask=(mask * 255).astype(np.uint8))
    blue_channel = water_region[:, :, 2]  # RGB format, blue is index 2
    blue_intensity = np.mean(blue_channel) / 255.0

    # Prepare input for U-Net
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, mask, blue_intensity

# --- 3. Main prediction function ---
def predict_uploaded_image(uploaded_file):
    # Step 1: Classify
    image_class = classify_uploaded_image(uploaded_file)
    print(f"📌 Predicted class: {image_class}")

    # Step 2: Prepare data
    img_array, mask, blue_intensity = prepare_uploaded_image(uploaded_file)

    # Step 3: Select correct model
    if image_class == 'lake':
        model = lake_model
    elif image_class == 'river':
        model = river_model
    elif image_class == 'harbor':
        model = harbor_model
    else:
        raise ValueError("Unknown class!")

    # Step 4: Predict
    prediction = model.predict(img_array)
    predicted_depth_class = np.argmax(prediction)

    depth_mapping = {0: 'Low Water Level', 1: 'Medium Water Level', 2: 'High Water Level'}
    predicted_depth = depth_mapping[predicted_depth_class]

    print(f"📊 Predicted Water Level: {predicted_depth}")

    return {
        "class": image_class,
        "predicted_depth": predicted_depth,
        "blue_intensity": blue_intensity
    }