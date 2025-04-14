import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
import re
import gc  # Garbage collector

# Ces variables seront déterminées automatiquement
IMAGE_HEIGHT = None
IMAGE_WIDTH = None
EPOCHS = 10
BATCH_SIZE = 8
SEQUENCE_LENGTH = 6  # Nombre d'images historiques à considérer
RESIZE_FACTOR = 4  # Facteur de réduction de la taille des images (4 = 1/4 de la taille originale)

# 1. Détection automatique de la taille d'image et chargement des données
def detect_image_size_and_load_dataset(base_dir, attribute_name):
    """
    Détecte automatiquement la taille d'image et charge le dataset
    """
    global IMAGE_HEIGHT, IMAGE_WIDTH
    
    print(f"Chargement des images {attribute_name}...")
    
    images = []
    dates = []
    
    attribute_dir = os.path.join(base_dir, attribute_name)
    
    # Obtenir la liste de tous les fichiers
    all_files = sorted([f for f in os.listdir(attribute_dir) if f.endswith('.png')])
    
    if not all_files:
        raise ValueError(f"Aucune image PNG trouvée dans {attribute_dir}")
    
    # Détecter la taille d'image à partir du premier fichier
    first_image_path = os.path.join(attribute_dir, all_files[0])
    with Image.open(first_image_path) as img:
        orig_width, orig_height = img.size
        # Réduire la taille des images pour économiser la mémoire
        IMAGE_WIDTH = orig_width // RESIZE_FACTOR
        IMAGE_HEIGHT = orig_height // RESIZE_FACTOR
        print(f"Taille d'image originale: {orig_width}x{orig_height}")
        print(f"Taille d'image redimensionnée: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    
    # Charger les images par lots pour économiser la mémoire
    batch_size = 100  # Traiter les images par lots de 100
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_images = []
        batch_dates = []
        
        for filename in batch_files:
            # Extraction de la date avec regex (format A20000101.png)
            date_match = re.match(r'A(\d{4})(\d{2})(\d{2})\.png', filename)
            
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                
                # Chargement de l'image PNG
                image_path = os.path.join(attribute_dir, filename)
                image = load_image(image_path)
                
                # Normalisation de l'image
                image = normalize_image(image)
                
                batch_images.append(image)
                batch_dates.append(datetime(year, month, day))
        
        # Ajouter le lot à la liste principale
        images.extend(batch_images)
        dates.extend(batch_dates)
        
        # Afficher la progression
        print(f"  Chargées {min(i+batch_size, len(all_files))}/{len(all_files)} images...")
        
        # Forcer la collecte des déchets pour libérer la mémoire
        gc.collect()
    
    # Tri des images par date
    sorted_data = sorted(zip(dates, images), key=lambda x: x[0])
    dates = [item[0] for item in sorted_data]
    images = [item[1] for item in sorted_data]
    
    print(f"Nombre total d'images chargées: {len(images)}")
    
    return np.array(images), np.array(dates)

def load_image(filepath):
    """
    Charge une image PNG, la redimensionne et la convertit en array numpy
    """
    try:
        with Image.open(filepath) as img:
            # Conversion en niveaux de gris si l'image est en couleur
            if img.mode != 'L':
                img = img.convert('L')
            
            # Redimensionnement pour économiser la mémoire
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            
            image_array = np.array(img).astype(np.float32)
            return image_array
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {filepath}: {e}")
        # Renvoyer une image noire en cas d'erreur
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

def normalize_image(image):
    """Normalise les valeurs de l'image entre 0 et 1"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

# 2. Création des séquences pour l'apprentissage temporel - version optimisée pour la mémoire
def create_temporal_sequences(images, dates):
    """
    Crée des séquences d'images pour l'apprentissage temporel.
    Utilise un générateur pour économiser la mémoire.
    """
    print(f"Préparation des séquences temporelles pour {len(images)} images...")
    
    # Calculer combien de séquences nous aurons
    num_sequences = len(images) - SEQUENCE_LENGTH
    print(f"Nombre de séquences à générer: {num_sequences}")
    
    # Créer des arrays vides pour stocker les métadonnées de séquence
    sequence_dates = []
    target_dates = []
    
    for i in range(num_sequences):
        sequence_date = dates[i:i+SEQUENCE_LENGTH]
        target_date = dates[i+SEQUENCE_LENGTH]
        
        sequence_dates.append(sequence_date)
        target_dates.append(target_date)
    
    # Définir un générateur pour créer les séquences à la demande pendant l'entraînement
    def sequence_generator():
        for i in range(num_sequences):
            # Séquence d'entrée: N images consécutives
            sequence = images[i:i+SEQUENCE_LENGTH]
            # Cible: l'image suivante
            target = images[i+SEQUENCE_LENGTH]
            
            # Ajouter dimension de canal pour le modèle
            sequence = np.expand_dims(sequence, axis=-1)
            target = np.expand_dims(target, axis=-1)
            
            yield sequence, target
    
    # Créer un dataset TensorFlow à partir du générateur
    output_signature = (
        tf.TensorSpec(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        sequence_generator,
        output_signature=output_signature
    )
    
    return dataset, num_sequences, np.array(sequence_dates), np.array(target_dates)

# 3. Création du modèle de deep learning
def create_attribute_model():
    """
    Crée un modèle ConvLSTM pour la prédiction des cartes d'attributs
    Version optimisée pour fonctionner avec des images plus petites
    """
    # Entrée: séquence d'images
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    inputs = layers.Input(input_shape)
    
    # Encodeur: extraction de caractéristiques spatiales et temporelles
    # Utilisation de filtres plus petits pour économiser la mémoire
    x = layers.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Décodeur: reconstruction de l'image de prédiction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    # Couche de sortie: prédiction de l'image
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# 4. Module pour gérer la conversion entre coordonnées et indices d'image
class CoordinateConverter:
    """
    Convertit entre coordonnées géographiques (lat/lon) et indices d'image
    """
    def __init__(self, lon_min, lon_max, lat_min, lat_max, width, height, original_width, original_height):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.width = width
        self.height = height
        self.original_width = original_width
        self.original_height = original_height
        
        # Facteurs de mise à l'échelle
        self.width_scale = width / original_width
        self.height_scale = height / original_height
    
    def coord_to_pixel(self, lat, lon):
        """Convertit lat/lon en indices de pixel dans l'image redimensionnée"""
        # Vérification des limites
        if lon < self.lon_min or lon > self.lon_max or lat < self.lat_min or lat > self.lat_max:
            raise ValueError(f"Coordonnées hors limites: lat={lat}, lon={lon}")
        
        # Conversion en coordonnées normalisées (0-1)
        x_norm = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        # Note: inversion pour la latitude (nord en haut)
        y_norm = 1.0 - (lat - self.lat_min) / (self.lat_max - self.lat_min)
        
        # Conversion en indices pour l'image redimensionnée
        x_idx = int(x_norm * (self.width - 1))
        y_idx = int(y_norm * (self.height - 1))
        
        return y_idx, x_idx
    
    def pixel_to_coord(self, y_idx, x_idx):
        """Convertit indices de pixel en lat/lon"""
        # Vérification des limites
        if x_idx < 0 or x_idx >= self.width or y_idx < 0 or y_idx >= self.height:
            raise ValueError(f"Indices hors limites: y_idx={y_idx}, x_idx={x_idx}")
        
        x_norm = x_idx / (self.width - 1)
        y_norm = y_idx / (self.height - 1)
        
        lon = self.lon_min + x_norm * (self.lon_max - self.lon_min)
        lat = self.lat_max - y_norm * (self.lat_max - self.lat_min)
        
        return lat, lon

# 5. Fonction d'inférence: prédiction pour des coordonnées spécifiques
def predict_attribute_value(model, historical_images, converter, lat, lon):
    """
    Prédit la valeur d'un attribut pour des coordonnées spécifiques
    """
    # Sélection des N dernières images pour la séquence
    input_sequence = historical_images[-SEQUENCE_LENGTH:]
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Ajout dimension batch
    input_sequence = np.expand_dims(input_sequence, axis=-1)  # Ajout dimension canal
    
    # Prédiction de la carte complète
    predicted_map = model.predict(input_sequence)[0, :, :, 0]
    
    # Extraction de la valeur aux coordonnées spécifiques
    y_idx, x_idx = converter.coord_to_pixel(lat, lon)
    predicted_value = predicted_map[y_idx, x_idx]
    
    return predicted_value, predicted_map

# 6. Visualisation des prédictions
def visualize_prediction(original, predicted, title="Comparaison Original vs Prédit"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Image originale
    im1 = ax1.imshow(original, cmap='viridis')
    ax1.set_title('Image Originale')
    ax1.set_axis_off()
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Image prédite
    im2 = ax2.imshow(predicted, cmap='viridis')
    ax2.set_title('Image Prédite')
    ax2.set_axis_off()
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
    plt.show()

# 7. Code principal d'exécution
def main():
    # Configurations des USA (ajustez selon vos images)
    usa_lon_min = -125.0
    usa_lon_max = -66.0
    usa_lat_min = 24.0
    usa_lat_max = 49.0
    
    # Chemin vers le dossier racine contenant les sous-dossiers d'attributs
    base_dir = "visualization_outputUp"
    attribute_name = "AvgSurfT"  # Choisir parmi les attributs disponibles
    
    # Chargement des données avec détection automatique de la taille d'image
    print(f"Chargement des données pour l'attribut {attribute_name}...")
    images, dates = detect_image_size_and_load_dataset(base_dir, attribute_name)
    
    # Sauvegarde de la taille originale avant redimensionnement
    original_width = IMAGE_WIDTH * RESIZE_FACTOR
    original_height = IMAGE_HEIGHT * RESIZE_FACTOR
    
    # Initialisation du convertisseur de coordonnées (après détection de la taille)
    converter = CoordinateConverter(
        usa_lon_min, usa_lon_max, usa_lat_min, usa_lat_max, 
        IMAGE_WIDTH, IMAGE_HEIGHT, original_width, original_height
    )
    
    # Création des séquences temporelles avec la méthode optimisée
    print("Création des séquences temporelles...")
    dataset, num_sequences, sequence_dates, target_dates = create_temporal_sequences(images, dates)
    print(f"Nombre de séquences: {num_sequences}")
    
    # Affichage des informations sur les données
    print(f"Période couverte: {dates[0]} à {dates[-1]}")
    print(f"Nombre total d'images: {len(images)}")
    
    # Préparation des données pour l'entraînement
    # Utilisation de tf.data pour un traitement par lots efficace
    train_size = int(0.8 * num_sequences)
    val_size = num_sequences - train_size
    
    # Création des ensembles d'entraînement et de validation
    train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"Taille de l'ensemble d'entraînement: {train_size} séquences")
    print(f"Taille de l'ensemble de validation: {val_size} séquences")
    
    # Création et entraînement du modèle
    print("Création du modèle...")
    model = create_attribute_model()
    print("Résumé du modèle:")
    model.summary()
    
    # Définition du nom du modèle et des fichiers associés
    model_name = f"{attribute_name}_model"
    checkpoint_path = f"{model_name}.h5"
    history_path = f"{model_name}_history.csv"
    
    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Entraînement du modèle
    print("Démarrage de l'entraînement...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Enregistrement de l'historique d'entraînement
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(history_path, index=False)
        print(f"Historique d'entraînement enregistré dans {history_path}")
        
        # Tracé des courbes d'apprentissage
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entraînement')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Fonction de perte')
        plt.xlabel('Époque')
        plt.ylabel('Erreur quadratique moyenne')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Entraînement')
        plt.plot(history.history['val_mae'], label='Validation')
        plt.title('Précision')
        plt.xlabel('Époque')
        plt.ylabel('Erreur absolue moyenne')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{model_name}_learning_curves.png", dpi=300)
        plt.show()
        
    except KeyboardInterrupt:
        print("Entraînement interrompu par l'utilisateur.")
    
    # Test de prédiction sur quelques points aux États-Unis
    test_locations = [
        (37.7749, -122.4194, "San Francisco"),
        (40.7128, -74.0060, "New York"),
        (29.7604, -95.3698, "Houston"),
        (41.8781, -87.6298, "Chicago")
    ]
    
    print("\nTests de prédiction:")
    for lat, lon, name in test_locations:
        try:
            predicted_value, _ = predict_attribute_value(
                model, images, converter, lat, lon
            )
            print(f"{name} (lat={lat}, lon={lon}): {predicted_value:.4f}")
        except ValueError as e:
            print(f"{name}: {e}")
    
    # Visualisation de la dernière image réelle vs prédiction
    last_real_image = images[-1]
    _, predicted_map = predict_attribute_value(
        model, images, converter, 37.7749, -122.4194  # San Francisco
    )
    
    visualize_prediction(
        last_real_image, 
        predicted_map, 
        f"Comparaison {attribute_name}: Réel vs Prédit"
    )
    
    # Sauvegarde du modèle entraîné
    model.save(f'{model_name}_full.h5')
    print(f"Modèle entraîné sauvegardé sous '{model_name}_full.h5'")

if __name__ == "__main__":
    main()