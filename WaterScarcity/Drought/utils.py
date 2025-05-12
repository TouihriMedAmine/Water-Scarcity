import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from werkzeug.datastructures import FileStorage

GRID_SHAPE = (88, 130)
FEATURES = ['Evap', 'Rainf', 'RootMoist', 'SoilM_0_10cm', 'TVeg']

def preprocess_single_images(files_dict):
    def read_image(img_file: FileStorage):
        img_file.seek(0)  # ensure pointer is at the start
        img_array = np.frombuffer(img_file.read(), np.uint8)

        if img_array.size == 0:
            raise ValueError(f"Uploaded image '{img_file.filename}' is empty or unreadable.")

        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {img_file.filename}")

        img_resized = cv2.resize(img, (GRID_SHAPE[1], GRID_SHAPE[0]))
        return img_resized.astype(np.float32) / 255.0

    layers = []
    for feature in FEATURES:
        if feature not in files_dict:
            raise ValueError(f"Missing input: {feature}")
        img = read_image(files_dict[feature])
        layers.append(img)  # shape: (88, 130)

    stacked = np.stack(layers, axis=-1)  # shape: (88, 130, 5)

    # Normalize per feature
    flat = stacked.reshape(-1, 5)
    scaled = MinMaxScaler().fit_transform(flat)
    normalized = scaled.reshape(88, 130, 5)
    

    return np.expand_dims(normalized, axis=0)  # shape: (1, 88, 130, 5)
