# -*- coding: utf-8 -*-
"""
Modèle principal pour l'optimisation d'irrigation

Ce module intègre tous les composants du modèle de Deep Learning pour
l'optimisation d'irrigation agricole, incluant les encodeurs spécifiques aux variables,
le module de fusion temporelle, les mécanismes d'attention et le système de recommandation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_ENTRAINEMENT, VARIABLES, CONFIG_PIPELINE
from deep_learning.encodeurs import get_encodeur
from deep_learning.fusion_temporelle import ModuleFusionTemporelle
from deep_learning.mecanismes_attention import ModuleAttention
from deep_learning.systeme_recommandation import SystemeRecommandation


class TeteClassification(nn.Module):
    """
    Tête de classification pour les tâches auxiliaires.
    """
    def __init__(self, dim_entree, dim_sortie, couches_cachees=None):
        """
        Initialise la tête de classification.
        
        Args:
            dim_entree (int): Dimension d'entrée
            dim_sortie (int): Dimension de sortie
            couches_cachees (list, optional): Liste des dimensions des couches cachées
        """
        super(TeteClassification, self).__init__()
        
        couches_cachees = couches_cachees or [128, 64]
        couches = []
        
        # Première couche
        couches.append(nn.Linear(dim_entree, couches_cachees[0]))
        couches.append(nn.ReLU())
        couches.append(nn.Dropout(0.3))
        
        # Couches cachées
        for i in range(len(couches_cachees) - 1):
            couches.append(nn.Linear(couches_cachees[i], couches_cachees[i+1]))
            couches.append(nn.ReLU())
            couches.append(nn.Dropout(0.3))
        
        # Couche de sortie
        couches.append(nn.Linear(couches_cachees[-1], dim_sortie))
        
        self.reseau = nn.Sequential(*couches)
    
    def forward(self, x):
        """
        Propagation avant de la tête de classification.
        
        Args:
            x (Tensor): Tensor d'entrée
            
        Returns:
            Tensor: Tensor de sortie
        """
        return self.reseau(x)


class TeteRegression(nn.Module):
    """
    Tête de régression pour les tâches de prédiction continue.
    """
    def __init__(self, dim_entree, dim_sortie, couches_cachees=None):
        """
        Initialise la tête de régression.
        
        Args:
            dim_entree (int): Dimension d'entrée
            dim_sortie (int): Dimension de sortie
            couches_cachees (list, optional): Liste des dimensions des couches cachées
        """
        super(TeteRegression, self).__init__()
        
        couches_cachees = couches_cachees or [128, 64]
        couches = []
        
        # Première couche
        couches.append(nn.Linear(dim_entree, couches_cachees[0]))
        couches.append(nn.ReLU())
        couches.append(nn.Dropout(0.3))
        
        # Couches cachées
        for i in range(len(couches_cachees) - 1):
            couches.append(nn.Linear(couches_cachees[i], couches_cachees[i+1]))
            couches.append(nn.ReLU())
            couches.append(nn.Dropout(0.3))
        
        # Couche de sortie
        couches.append(nn.Linear(couches_cachees[-1], dim_sortie))
        
        self.reseau = nn.Sequential(*couches)
    
    def forward(self, x):
        """
        Propagation avant de la tête de régression.
        
        Args:
            x (Tensor): Tensor d'entrée
            
        Returns:
            Tensor: Tensor de sortie
        """
        return self.reseau(x)


class ModeleIrrigationOptimisation(nn.Module):
    """
    Modèle complet pour l'optimisation d'irrigation.
    """
    def __init__(self, variables=None):
        """
        Initialise le modèle d'optimisation d'irrigation.
        
        Args:
            variables (list, optional): Liste des variables à utiliser
        """
        super(ModeleIrrigationOptimisation, self).__init__()
        
        # Variables à utiliser
        self.variables = variables or list(VARIABLES.keys())
        
        # Dimension des caractéristiques
        self.dim_features = 256
        
        # Créer les encodeurs spécifiques aux variables
        self.encodeurs = nn.ModuleDict()
        for var in self.variables:
            self.encodeurs[var] = get_encodeur(var)
        
        # Module d'attention
        self.module_attention = ModuleAttention(dim_entree=self.dim_features)
        
        # Module de fusion temporelle
        self.module_fusion = ModuleFusionTemporelle(dim_entree=self.dim_features)
        
        # Têtes de prédiction pour les différentes tâches
        # Tâche principale: prédiction de la quantité d'irrigation
        self.tete_irrigation = TeteRegression(self.dim_features, 1)
        
        # Tâches auxiliaires
        self.tete_humidite_sol = TeteRegression(self.dim_features, 1)
        self.tete_evapotranspiration = TeteRegression(self.dim_features, 1)
        self.tete_stress_hydrique = TeteRegression(self.dim_features, 1)
        
        # Classification du type de sol (5 classes: sableux, limoneux, argileux, loam, organique)
        self.tete_type_sol = TeteClassification(self.dim_features, 5)
        
        # Prédiction d'adéquation des cultures (8 cultures)
        self.tete_adequation_culture = TeteRegression(self.dim_features, 8)
    
    def forward(self, x):
        """
        Propagation avant du modèle complet.
        
        Args:
            x (dict): Dictionnaire des séquences d'images pour chaque variable
                     {var_name: tensor de forme [B, T, C, H, W]}
            
        Returns:
            dict: Dictionnaire des prédictions pour chaque tâche
        """
        # Extraire les caractéristiques de chaque variable
        caracteristiques = {}
        for var in self.variables:
            if var in x:
                caracteristiques[var] = self.encodeurs[var](x[var])
        
        # Appliquer l'attention croisée et temporelle
        caracteristiques_attention, _ = self.module_attention(caracteristiques)
        
        # Fusionner les caractéristiques de toutes les variables
        caracteristiques_fusionnees = torch.stack([caracteristiques_attention[var] for var in self.variables], dim=2)
        caracteristiques_fusionnees = torch.mean(caracteristiques_fusionnees, dim=2)  # [B, T, D]
        
        # Appliquer la fusion temporelle
        caracteristiques_finales = self.module_fusion(caracteristiques_fusionnees)  # [B, T, D]
        
        # Utiliser les caractéristiques du dernier pas de temps pour les prédictions
        caracteristiques_finales = caracteristiques_finales[:, -1, :]  # [B, D]
        
        # Prédictions pour chaque tâche
        predictions = {
            'irrigation': self.tete_irrigation(caracteristiques_finales),
            'humidite_sol': self.tete_humidite_sol(caracteristiques_finales),
            'evapotranspiration': self.tete_evapotranspiration(caracteristiques_finales),
            'stress_hydrique': self.tete_stress_hydrique(caracteristiques_finales),
            'type_sol': self.tete_type_sol(caracteristiques_finales),
            'adequation_culture': self.tete_adequation_culture(caracteristiques_finales)
        }
        
        return predictions
    
    def calculer_perte(self, predictions, cibles):
        """
        Calcule la perte totale pour toutes les tâches.
        
        Args:
            predictions (dict): Dictionnaire des prédictions
            cibles (dict): Dictionnaire des valeurs cibles
            
        Returns:
            Tensor: Perte totale
        """
        # Poids des différentes tâches
        poids = CONFIG_ENTRAINEMENT['poids_taches']
        
        # Fonctions de perte
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        
        # Calculer les pertes individuelles
        pertes = {}
        
        # Tâche principale: irrigation
        if 'irrigation' in cibles:
            pertes['irrigation'] = mse(predictions['irrigation'], cibles['irrigation'])
        
        # Tâches auxiliaires
        if 'humidite_sol' in cibles:
            pertes['humidite_sol'] = mse(predictions['humidite_sol'], cibles['humidite_sol'])
        
        if 'evapotranspiration' in cibles:
            pertes['evapotranspiration'] = mse(predictions['evapotranspiration'], cibles['evapotranspiration'])
        
        if 'stress_hydrique' in cibles:
            pertes['stress_hydrique'] = mse(predictions['stress_hydrique'], cibles['stress_hydrique'])
        
        if 'type_sol' in cibles:
            pertes['type_sol'] = ce(predictions['type_sol'], cibles['type_sol'])
        
        if 'adequation_culture' in cibles:
            pertes['adequation_culture'] = mse(predictions['adequation_culture'], cibles['adequation_culture'])
        
        # Calculer la perte totale pondérée
        perte_totale = sum(poids.get(tache, 0) * pertes.get(tache, 0) for tache in pertes)
        
        return perte_totale, pertes
    
    def generer_recommandations(self, predictions, latitude, longitude, date, utilisation_actuelle=None, contraintes_eau=None):
        """
        Génère des recommandations basées sur les prédictions du modèle.
        
        Args:
            predictions (dict): Prédictions du modèle
            latitude (float): Latitude des coordonnées géographiques
            longitude (float): Longitude des coordonnées géographiques
            date (datetime): Date pour laquelle faire les recommandations
            utilisation_actuelle (dict, optional): Informations sur l'utilisation actuelle des terres
            contraintes_eau (dict, optional): Contraintes d'eau disponible
            
        Returns:
            dict: Recommandations complètes
        """
        # Convertir les prédictions en format adapté pour le système de recommandation
        predictions_formattees = self._formater_predictions(predictions)
        
        # Créer le système de recommandation
        systeme_recommandation = SystemeRecommandation()
        
        # Générer les recommandations
        recommandations = systeme_recommandation.generer_recommandations(
            latitude=latitude,
            longitude=longitude,
            date=date,
            predictions=predictions_formattees,
            utilisation_actuelle=utilisation_actuelle,
            contraintes_eau=contraintes_eau
        )
        
        return recommandations
    
    def _formater_predictions(self, predictions):
        """
        Formate les prédictions du modèle pour le système de recommandation.
        
        Args:
            predictions (dict): Prédictions brutes du modèle
            
        Returns:
            dict: Prédictions formatées
        """
        # Convertir les tensors en valeurs numériques
        predictions_formattees = {}
        
        # Irrigation (litres/m²)
        if 'irrigation' in predictions:
            predictions_formattees['irrigation'] = predictions['irrigation'].item() if hasattr(predictions['irrigation'], 'item') else float(predictions['irrigation'])
        
        # Humidité du sol (0-1)
        if 'humidite_sol' in predictions:
            predictions_formattees['humidite_sol'] = predictions['humidite_sol'].item() if hasattr(predictions['humidite_sol'], 'item') else float(predictions['humidite_sol'])
        
        # Évapotranspiration (mm/jour)
        if 'evapotranspiration' in predictions:
            predictions_formattees['evapotranspiration'] = predictions['evapotranspiration'].item() if hasattr(predictions['evapotranspiration'], 'item') else float(predictions['evapotranspiration'])
        
        # Stress hydrique (0-1)
        if 'stress_hydrique' in predictions:
            predictions_formattees['stress_hydrique'] = predictions['stress_hydrique'].item() if hasattr(predictions['stress_hydrique'], 'item') else float(predictions['stress_hydrique'])
        
        # Type de sol (classification)
        if 'type_sol' in predictions:
            types_sol = ['sableux', 'limoneux', 'argileux', 'loam', 'organique']
            if hasattr(predictions['type_sol'], 'argmax'):
                idx = predictions['type_sol'].argmax().item()
            else:
                idx = np.argmax(predictions['type_sol'])
            predictions_formattees['type_sol'] = types_sol[idx]
        
        # Adéquation des cultures (scores 0-1 pour chaque culture)
        if 'adequation_culture' in predictions:
            cultures = ['maïs', 'blé', 'riz', 'soja', 'tomate', 'pomme de terre', 'coton', 'canne à sucre']
            scores = predictions['adequation_culture']
            if hasattr(scores, 'detach'):
                scores = scores.detach().cpu().numpy()
            predictions_formattees['adequation_cultures'] = {culture: float(score) for culture, score in zip(cultures, scores)}
        
        return predictions_formattees


def entrainer_modele(modele, train_loader, val_loader, epoques=None, patience=None, taux_apprentissage=None):
    """
    Entraîne le modèle d'optimisation d'irrigation.
    
    Args:
        modele (ModeleIrrigationOptimisation): Modèle à entraîner
        train_loader (DataLoader): DataLoader pour les données d'entraînement
        val_loader (DataLoader): DataLoader pour les données de validation
        epoques (int, optional): Nombre d'époques d'entraînement
        patience (int, optional): Patience pour l'early stopping
        taux_apprentissage (float, optional): Taux d'apprentissage
        
    Returns:
        ModeleIrrigationOptimisation: Modèle entraîné
    """
    # Utiliser les valeurs par défaut de la configuration si non spécifiées
    epoques = epoques or CONFIG_ENTRAINEMENT['epoques']
    patience = patience or CONFIG_ENTRAINEMENT['patience']
    taux_apprentissage = taux_apprentissage or CONFIG_ENTRAINEMENT['taux_apprentissage']
    
    # Définir l'optimiseur
    optimiseur = torch.optim.Adam(modele.parameters(), lr=taux_apprentissage)
    
    # Définir le scheduler pour réduire le taux d'apprentissage
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, mode='min', factor=0.5, patience=patience//2, verbose=True)
    
    # Initialiser les variables pour l'early stopping
    meilleure_perte_val = float('inf')
    compteur_patience = 0
    
    # Historique des pertes
    historique = {'train': [], 'val': []}
    
    # Boucle d'entraînement
    for epoque in range(epoques):
        # Mode entraînement
        modele.train()
        perte_train_totale = 0
        
        for batch_idx, (sequences, _) in enumerate(train_loader):
            # Mettre les données sur le GPU si disponible
            if torch.cuda.is_available():
                sequences = {var: seq.cuda() for var, seq in sequences.items()}
            
            # Réinitialiser les gradients
            optimiseur.zero_grad()
            
            # Propagation avant
            predictions = modele(sequences)
            
            # Calculer la perte (ici, nous utilisons des cibles fictives pour l'exemple)
            # Dans un cas réel, les cibles seraient extraites des données
            cibles = {
                'irrigation': torch.rand_like(predictions['irrigation']),
                'humidite_sol': torch.rand_like(predictions['humidite_sol']),
                'evapotranspiration': torch.rand_like(predictions['evapotranspiration']),
                'stress_hydrique': torch.rand_like(predictions['stress_hydrique']),
                'type_sol': torch.randint(0, 5, (predictions['type_sol'].size(0),)),
                'adequation_culture': torch.rand_like(predictions['adequation_culture'])
            }
            
            perte, _ = modele.calculer_perte(predictions, cibles)
            
            # Rétropropagation
            perte.backward()
            optimiseur.step()
            
            # Accumuler la perte
            perte_train_totale += perte.item()
        
        # Calculer la perte moyenne d'entraînement
        perte_train_moyenne = perte_train_totale / len(train_loader)
        historique['train'].append(perte_train_moyenne)
        
        # Mode évaluation
        modele.eval()
        perte_val_totale = 0
        
        with torch.no_grad():
            for sequences, _ in val_loader:
                # Mettre les données sur le GPU si disponible
                if torch.cuda.is_available():
                    sequences = {var: seq.cuda() for var, seq in sequences.items()}
                
                # Propagation avant
                predictions = modele(sequences)
                
                # Calculer la perte (ici, nous utilisons des cibles fictives pour l'exemple)
                cibles = {
                    'irrigation': torch.rand_like(predictions['irrigation']),
                    'humidite_sol': torch.rand_like(predictions['humidite_sol']),
                    'evapotranspiration': torch.rand_like(predictions['evapotranspiration']),
                    'stress_hydrique': torch.rand_like(predictions['stress_hydrique']),
                    'type_sol': torch.randint(0, 5, (predictions['type_sol'].size(0),)),
                    'adequation_culture': torch.rand_like(predictions['adequation_culture'])
                }
                
                perte, _ = modele.calculer_perte(predictions, cibles)
                
                # Accumuler la perte
                perte_val_totale += perte.item()
        
        # Calculer la perte moyenne de validation
        perte_val_moyenne = perte_val_totale / len(val_loader)
        historique['val'].append(perte_val_moyenne)
        
        # Mettre à jour le scheduler
        scheduler.step(perte_val_moyenne)
        
        # Afficher les progrès
        print(f"Époque {epoque+1}/{epoques} | Perte train: {perte_train_moyenne:.4f} | Perte val: {perte_val_moyenne:.4f}")
        
        # Early stopping
        if perte_val_moyenne < meilleure_perte_val:
            meilleure_perte_val = perte_val_moyenne
            compteur_patience = 0
            
            # Sauvegarder le meilleur modèle
            torch.save(modele.state_dict(), os.path.join(CHEMIN_DONNEES['modeles'], 'meilleur_modele.pth'))
        else:
            compteur_patience += 1
            if compteur_patience >= patience:
                print(f"Early stopping à l'époque {epoque+1}")
                break
    
    # Charger le meilleur modèle
    modele.load_state_dict(torch.load(os.path.join(CHEMIN_DONNEES['modeles'], 'meilleur_modele.pth')))
    
    return modele, historique


def evaluer_modele(modele, test_loader):
    """
    Évalue le modèle sur l'ensemble de test.
    
    Args:
        modele (ModeleIrrigationOptimisation): Modèle à évaluer
        test_loader (DataLoader): DataLoader pour les données de test
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Mode évaluation
    modele.eval()
    
    # Initialiser les métriques
    metriques = {
        'mse_irrigation': 0,
        'mse_humidite_sol': 0,
        'mse_evapotranspiration': 0,
        'mse_stress_hydrique': 0,
        'precision_type_sol': 0,
        'mse_adequation_culture': 0
    }
    
    with torch.no_grad():
        for sequences, _ in test_loader:
            # Mettre les données sur le GPU si disponible
            if torch.cuda.is_available():
                sequences = {var: seq.cuda() for var, seq in sequences.items()}
            
            # Propagation avant
            predictions = modele(sequences)
            
            # Calculer les métriques (ici, nous utilisons des cibles fictives pour l'exemple)
            cibles = {
                'irrigation': torch.rand_like(predictions['irrigation']),
                'humidite_sol': torch.rand_like(predictions['humidite_sol']),
                'evapotranspiration': torch.rand_like(predictions['evapotranspiration']),
                'stress_hydrique': torch.rand_like(predictions['stress_hydrique']),
                'type_sol': torch.randint(0, 5, (predictions['type_sol'].size(0),)),
                'adequation_culture': torch.rand_like(predictions['adequation_culture'])
            }
            
            # MSE pour les tâches de régression
            metriques['mse_irrigation'] += F.mse_loss(predictions['irrigation'], cibles['irrigation']).item()
            metriques['mse_humidite_sol'] += F.mse_loss(predictions['humidite_sol'], cibles['humidite_sol']).item()
            metriques['mse_evapotranspiration'] += F.mse_loss(predictions['evapotranspiration'], cibles['evapotranspiration']).item()
            metriques['mse_stress_hydrique'] += F.mse_loss(predictions['stress_hydrique'], cibles['stress_hydrique']).item()
            metriques['mse_adequation_culture'] += F.mse_loss(predictions['adequation_culture'], cibles['adequation_culture']).item()
            
            # Précision pour la classification du type de sol
            _, predicted = torch.max(predictions['type_sol'], 1)
            metriques['precision_type_sol'] += (predicted == cibles['type_sol']).sum().item() / cibles['type_sol'].size(0)
    
    # Calculer les moyennes
    for metrique in metriques:
        metriques[metrique] /= len(test_loader)
    
    return metriques


if __name__ == "__main__":
    # Test du modèle
    print("Test du modèle d'optimisation d'irrigation...")
    
    # Créer le modèle
    modele = ModeleIrrigationOptimisation()
    
    # Afficher l'architecture du modèle
    print(modele)
    
    # Créer des données de test
    batch_size = 2
    seq_len = 10
    channels = 3
    height = 224
    width = 224
    
    # Créer des séquences d'images pour chaque variable
    sequences_test = {}
    for var in VARIABLES.keys():
        sequences_test[var] = torch.randn(batch_size, seq_len, channels, height, width)
    
    # Propagation avant
    predictions = modele(sequences_test)
    
    # Afficher les formes des prédictions
    print("\nFormes des prédictions:")
    for tache, pred in predictions.items():
        print(f"  {tache}: {pred.shape}")
    
    # Tester la génération de recommandations
    print("\nTest de la génération de recommandations...")
    recommandations = modele.generer_recommandations(
        predictions=modele._formater_predictions(predictions),
        latitude=34.05,
        longitude=-118.25,
        date=datetime(2023, 5, 15),
        utilisation_actuelle={'culture': 'maïs', 'superficie': 10.0},
        contraintes_eau={'disponibilite': 'limitée', 'source': 'puits'}
    )
    
    # Afficher les recommandations
    import json
    print(json.dumps(recommandations, indent=2, ensure_ascii=False))