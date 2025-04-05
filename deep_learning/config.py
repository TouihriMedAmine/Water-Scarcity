# -*- coding: utf-8 -*-
"""
Configuration pour le modèle de Deep Learning d'optimisation d'irrigation

Ce fichier contient tous les paramètres de configuration pour le modèle de Deep Learning
utilisé dans l'optimisation de l'irrigation agricole.
"""

# Chemins des données
CHEMIN_DONNEES = {
    "images": "../visualization_output",
    "modeles": "./modeles",
    "resultats": "./resultats"
}

# Variables environnementales à utiliser dans le modèle
VARIABLES = {
    # Consommation d'eau par les plantes
    "Evap_tavg": {"titre": "Évapotranspiration", "type": "consommation"},
    "Tveg_tavg": {"titre": "Transpiration des plantes", "type": "consommation"},
    "ESoil_tavg": {"titre": "Évaporation du sol", "type": "consommation"},
    "PotEvap_tavg": {"titre": "Évaporation potentielle", "type": "consommation"},
    
    # Rétention d'eau dans le sol
    "SoilMoist_S_tavg": {"titre": "Humidité du sol", "type": "retention"},
    "SoilMoi0_10cm_inst": {"titre": "Humidité du sol (0-10cm)", "type": "retention"},
    "RootMoist_inst": {"titre": "Humidité de la zone racinaire", "type": "retention"},
    
    # Demande d'irrigation
    "Tair_f_inst": {"titre": "Température de l'air", "type": "demande"},
    "SWdown_f_tavg": {"titre": "Rayonnement solaire", "type": "demande"},
    "Rainf_tavg": {"titre": "Précipitations", "type": "demande"}
}

# Configuration du pipeline de données
CONFIG_PIPELINE = {
    "taille_sequence": 30,  # Nombre de jours dans une séquence
    "pas_sequence": 7,     # Décalage entre séquences consécutives
    "taille_lot": 32,      # Taille du batch pour l'entraînement
    "proportion_validation": 0.2,  # Proportion des données pour la validation
    "proportion_test": 0.1,       # Proportion des données pour le test
    "resolution_image": (224, 224)  # Résolution des images après redimensionnement
}

# Configuration des encodeurs spécifiques aux variables
CONFIG_ENCODEURS = {
    "ESoil_tavg": {
        "type": "3D_CNN",
        "filtres": [32, 64, 128],
        "taille_noyau": (3, 3, 3),
        "connexions_residuelles": True
    },
    "RootMoist_inst": {
        "type": "U_Net",
        "filtres": [32, 64, 128, 256],
        "connexions_skip": True
    },
    "Tair_f_inst": {
        "type": "ViT",
        "taille_patch": 16,
        "dim_embedding": 256,
        "profondeur": 6,
        "tetes_attention": 8
    },
    "Rainf_tavg": {
        "type": "TemporalConvNet",
        "filtres": [32, 64, 128],
        "taille_noyau": 3,
        "dilatation": [1, 2, 4]
    },
    "default": {
        "type": "CNN",
        "filtres": [32, 64, 128],
        "taille_noyau": 3
    }
}

# Configuration du module de fusion temporelle
CONFIG_FUSION_TEMPORELLE = {
    "type_convlstm": {
        "filtres": [64, 128],
        "taille_noyau": (3, 3),
        "dropout": 0.3
    },
    "type_transformer": {
        "dim_modele": 256,
        "tetes_attention": 8,
        "profondeur": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1
    }
}

# Configuration des mécanismes d'attention
CONFIG_ATTENTION = {
    "attention_croisee": {
        "dim_cle": 64,
        "dim_valeur": 64,
        "tetes": 8
    },
    "attention_temporelle": {
        "dim_cle": 64,
        "fenetres": [7, 14, 30]
    },
    "attention_spatiale": {
        "canaux": 128,
        "reduction": 16
    }
}

# Configuration de l'entraînement
CONFIG_ENTRAINEMENT = {
    "epoques": 100,
    "patience": 10,  # Pour l'early stopping
    "taux_apprentissage": 1e-4,
    "poids_taches": {
        "irrigation": 0.5,
        "humidite_sol": 0.2,
        "evapotranspiration": 0.1,
        "stress_hydrique": 0.1,
        "type_sol": 0.05,
        "adequation_culture": 0.05
    }
}

# Configuration du système de recommandation
CONFIG_RECOMMANDATION = {
    "seuils_stress_hydrique": {
        "faible": 0.3,
        "modere": 0.6,
        "eleve": 0.8
    },
    "methodes_irrigation": [
        "goutte-à-goutte",
        "aspersion",
        "inondation"
    ],
    "cultures": [
        "maïs", "blé", "riz", "soja", "tomate", 
        "pomme de terre", "coton", "canne à sucre"
    ]
}