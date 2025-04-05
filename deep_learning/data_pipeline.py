# -*- coding: utf-8 -*-
"""
Pipeline de données pour le modèle de Deep Learning d'optimisation d'irrigation

Ce module gère le chargement, le prétraitement et la structuration des données
pour l'entraînement du modèle de Deep Learning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import sys

# Ajouter le répertoire parent au chemin pour pouvoir importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_PIPELINE, VARIABLES, CHEMIN_DONNEES


class SequenceTemporelleDataset(Dataset):
    """
    Dataset pour gérer les séquences temporelles d'images satellitaires.
    Chaque élément est une séquence de 'taille_sequence' jours consécutifs
    pour toutes les variables environnementales.
    """
    
    def __init__(self, repertoire_base, liste_dates, variables=None, taille_sequence=30, 
                 transform=None, mode='train'):
        """
        Initialise le dataset de séquences temporelles.
        
        Args:
            repertoire_base (str): Chemin vers le répertoire contenant les images
            liste_dates (list): Liste des dates disponibles (format: 'AAAAMMJJ')
            variables (list): Liste des variables à inclure (si None, toutes les variables sont utilisées)
            taille_sequence (int): Nombre d'images consécutives dans une séquence
            transform (callable, optional): Transformations à appliquer aux images
            mode (str): 'train', 'val' ou 'test'
        """
        self.repertoire_base = repertoire_base
        self.liste_dates = sorted(liste_dates)
        self.variables = variables if variables else list(VARIABLES.keys())
        self.taille_sequence = taille_sequence
        self.transform = transform
        self.mode = mode
        
        # Créer les séquences valides (celles qui ont taille_sequence jours consécutifs)
        self.sequences_valides = self._creer_sequences_valides()
        
        print(f"Mode {mode}: {len(self.sequences_valides)} séquences créées")
    
    def _creer_sequences_valides(self):
        """
        Crée des séquences valides de dates consécutives.
        
        Returns:
            list: Liste des indices de début de séquences valides
        """
        sequences = []
        for i in range(len(self.liste_dates) - self.taille_sequence + 1):
            # Vérifier si les dates sont consécutives
            dates_sequence = self.liste_dates[i:i+self.taille_sequence]
            if self._est_sequence_consecutive(dates_sequence):
                sequences.append(i)
        return sequences
    
    def _est_sequence_consecutive(self, dates):
        """
        Vérifie si une séquence de dates est consécutive.
        
        Args:
            dates (list): Liste de dates au format 'AAAAMMJJ'
            
        Returns:
            bool: True si les dates sont consécutives, False sinon
        """
        # Convertir les dates en objets datetime pour faciliter la comparaison
        from datetime import datetime
        dates_dt = [datetime.strptime(date, 'A%Y%m%d') for date in dates]
        
        # Vérifier si chaque paire de dates consécutives a un écart d'un jour
        for i in range(len(dates_dt) - 1):
            diff = (dates_dt[i+1] - dates_dt[i]).days
            if diff != 1:
                return False
        return True
    
    def __len__(self):
        return len(self.sequences_valides)
    
    def __getitem__(self, idx):
        """
        Récupère une séquence d'images pour toutes les variables.
        
        Args:
            idx (int): Indice de la séquence
            
        Returns:
            dict: Dictionnaire contenant les séquences d'images pour chaque variable
                  et les métadonnées associées
        """
        debut_sequence = self.sequences_valides[idx]
        dates_sequence = self.liste_dates[debut_sequence:debut_sequence+self.taille_sequence]
        
        # Initialiser le dictionnaire de séquences
        sequences = {}
        
        # Charger les images pour chaque variable et chaque date
        for var in self.variables:
            sequences[var] = []
            for date in dates_sequence:
                chemin_image = os.path.join(self.repertoire_base, var, f"{date}.png")
                try:
                    image = Image.open(chemin_image)
                    if self.transform:
                        image = self.transform(image)
                    sequences[var].append(image)
                except FileNotFoundError:
                    print(f"Image non trouvée: {chemin_image}")
                    # Utiliser une image noire en cas d'image manquante
                    if self.transform:
                        image = self.transform(Image.new('RGB', (224, 224), color='black'))
                    else:
                        image = np.zeros((224, 224, 3), dtype=np.uint8)
                    sequences[var].append(image)
            
            # Convertir la liste d'images en tensor
            sequences[var] = torch.stack(sequences[var]) if isinstance(sequences[var][0], torch.Tensor) else np.array(sequences[var])
        
        # Ajouter les métadonnées
        metadata = {
            'dates': dates_sequence,
            'debut_sequence': debut_sequence
        }
        
        return sequences, metadata


def creer_dataloaders(repertoire_images=None, taille_sequence=None, taille_lot=None, 
                      proportion_validation=None, proportion_test=None):
    """
    Crée les dataloaders pour l'entraînement, la validation et le test.
    
    Args:
        repertoire_images (str, optional): Chemin vers le répertoire des images
        taille_sequence (int, optional): Nombre d'images dans une séquence
        taille_lot (int, optional): Taille du batch
        proportion_validation (float, optional): Proportion des données pour la validation
        proportion_test (float, optional): Proportion des données pour le test
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Utiliser les valeurs par défaut de la configuration si non spécifiées
    repertoire_images = repertoire_images or CHEMIN_DONNEES['images']
    taille_sequence = taille_sequence or CONFIG_PIPELINE['taille_sequence']
    taille_lot = taille_lot or CONFIG_PIPELINE['taille_lot']
    proportion_validation = proportion_validation or CONFIG_PIPELINE['proportion_validation']
    proportion_test = proportion_test or CONFIG_PIPELINE['proportion_test']
    
    # Définir les transformations pour les images
    transform = transforms.Compose([
        transforms.Resize(CONFIG_PIPELINE['resolution_image']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Récupérer la liste des dates disponibles (noms des fichiers sans extension)
    dates = set()
    for var in VARIABLES.keys():
        var_dir = os.path.join(repertoire_images, var)
        if os.path.exists(var_dir):
            for fichier in os.listdir(var_dir):
                if fichier.endswith('.png'):
                    dates.add(fichier[:-4])  # Enlever l'extension .png
    
    dates = sorted(list(dates))
    print(f"Nombre total de dates disponibles: {len(dates)}")
    
    # Diviser les dates en ensembles d'entraînement, validation et test
    train_dates, temp_dates = train_test_split(dates, test_size=proportion_validation+proportion_test, random_state=42)
    val_size = proportion_validation / (proportion_validation + proportion_test)
    val_dates, test_dates = train_test_split(temp_dates, test_size=1-val_size, random_state=42)
    
    print(f"Répartition des dates: {len(train_dates)} entraînement, {len(val_dates)} validation, {len(test_dates)} test")
    
    # Créer les datasets
    train_dataset = SequenceTemporelleDataset(
        repertoire_images, train_dates, taille_sequence=taille_sequence, transform=transform, mode='train')
    val_dataset = SequenceTemporelleDataset(
        repertoire_images, val_dates, taille_sequence=taille_sequence, transform=transform, mode='val')
    test_dataset = SequenceTemporelleDataset(
        repertoire_images, test_dates, taille_sequence=taille_sequence, transform=transform, mode='test')
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=taille_lot, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=taille_lot, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=taille_lot, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def visualiser_sequence(sequence, variables=None, indices_temps=None):
    """
    Visualise une séquence d'images pour les variables spécifiées.
    
    Args:
        sequence (dict): Dictionnaire contenant les séquences d'images pour chaque variable
        variables (list, optional): Liste des variables à visualiser
        indices_temps (list, optional): Liste des indices temporels à visualiser
    """
    variables = variables or list(sequence.keys())[:3]  # Limiter à 3 variables par défaut
    n_vars = len(variables)
    
    # Déterminer les indices temporels à afficher
    if indices_temps is None:
        taille_seq = sequence[variables[0]].shape[0]
        indices_temps = [0, taille_seq//2, taille_seq-1]  # Début, milieu, fin
    n_temps = len(indices_temps)
    
    # Créer la figure
    fig, axes = plt.subplots(n_vars, n_temps, figsize=(n_temps*4, n_vars*3))
    
    # Afficher chaque image
    for i, var in enumerate(variables):
        for j, t in enumerate(indices_temps):
            ax = axes[i, j] if n_vars > 1 else axes[j]
            img = sequence[var][t]
            
            # Convertir le tensor en numpy si nécessaire
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                # Dénormaliser
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"{VARIABLES[var]['titre']} (t={t})")
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def indexation_geospatiale(latitude, longitude, resolution_image):
    """
    Convertit des coordonnées géographiques en indices d'image.
    
    Args:
        latitude (float): Latitude en degrés
        longitude (float): Longitude en degrés
        resolution_image (tuple): Résolution de l'image (hauteur, largeur)
        
    Returns:
        tuple: (ligne, colonne) correspondant aux coordonnées dans l'image
    """
    # Cette fonction est une implémentation simplifiée et doit être adaptée
    # aux coordonnées réelles des images utilisées dans le projet
    
    # Exemple avec une carte mondiale simple
    hauteur, largeur = resolution_image
    
    # Normaliser les coordonnées
    # Longitude: -180 à 180 -> 0 à largeur
    col = int((longitude + 180) / 360 * largeur)
    
    # Latitude: 90 à -90 -> 0 à hauteur
    row = int((90 - latitude) / 180 * hauteur)
    
    return row, col


if __name__ == "__main__":
    # Test du pipeline de données
    print("Test du pipeline de données...")
    train_loader, val_loader, test_loader = creer_dataloaders()
    
    # Afficher un exemple de séquence
    for sequences, metadata in train_loader:
        print(f"Forme des séquences:")
        for var, seq in sequences.items():
            print(f"  {var}: {seq.shape}")
        
        print(f"Métadonnées: {metadata}")
        
        # Visualiser la première séquence du batch
        fig = visualiser_sequence({var: seq[0] for var, seq in sequences.items()})
        plt.savefig(os.path.join(CHEMIN_DONNEES['resultats'], 'exemple_sequence.png'))
        plt.close(fig)
        
        break  # Une seule itération pour le test