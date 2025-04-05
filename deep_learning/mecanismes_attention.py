# -*- coding: utf-8 -*-
"""
Mécanismes d'attention pour le modèle de Deep Learning d'optimisation d'irrigation

Ce module implémente différents mécanismes d'attention utilisés dans le modèle:
- Attention croisée entre variables
- Attention temporelle pour les périodes critiques
- Attention spatiale pour les prédictions spécifiques aux coordonnées
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ajouter le répertoire parent au chemin pour pouvoir importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_ATTENTION


class AttentionCroiseeVariables(nn.Module):
    """
    Module d'attention croisée entre différentes variables environnementales.
    Permet au modèle d'apprendre les relations entre les différentes variables.
    """
    def __init__(self, dim_entree, config=None):
        """
        Initialise le module d'attention croisée.
        
        Args:
            dim_entree (int): Dimension des caractéristiques d'entrée
            config (dict, optional): Configuration du module
        """
        super(AttentionCroiseeVariables, self).__init__()
        config = config or CONFIG_ATTENTION['attention_croisee']
        
        self.dim_cle = config['dim_cle']
        self.dim_valeur = config['dim_valeur']
        self.tetes = config['tetes']
        
        # Projections pour les clés, valeurs et requêtes
        self.projection_cle = nn.Linear(dim_entree, self.dim_cle * self.tetes)
        self.projection_valeur = nn.Linear(dim_entree, self.dim_valeur * self.tetes)
        self.projection_requete = nn.Linear(dim_entree, self.dim_cle * self.tetes)
        
        # Projection de sortie
        self.projection_sortie = nn.Linear(self.dim_valeur * self.tetes, dim_entree)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(dim_entree)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, caracteristiques_variables):
        """
        Propagation avant du module d'attention croisée.
        
        Args:
            caracteristiques_variables (dict): Dictionnaire des caractéristiques par variable
                                              {var_name: tensor de forme [B, T, D]}
            
        Returns:
            dict: Dictionnaire des caractéristiques mises à jour
        """
        # Extraire les noms des variables et leurs caractéristiques
        noms_variables = list(caracteristiques_variables.keys())
        batch_size = caracteristiques_variables[noms_variables[0]].size(0)
        seq_len = caracteristiques_variables[noms_variables[0]].size(1)
        dim_entree = caracteristiques_variables[noms_variables[0]].size(2)
        
        # Créer un dictionnaire pour stocker les caractéristiques mises à jour
        caracteristiques_mises_a_jour = {}
        
        # Pour chaque variable (comme requête)
        for var_requete in noms_variables:
            # Extraire les caractéristiques de la variable requête
            requete = caracteristiques_variables[var_requete]  # [B, T, D]
            
            # Projeter la requête
            requete_projetee = self.projection_requete(requete)  # [B, T, H*K]
            requete_projetee = requete_projetee.view(batch_size, seq_len, self.tetes, self.dim_cle)
            requete_projetee = requete_projetee.permute(0, 2, 1, 3)  # [B, H, T, K]
            
            # Initialiser le résultat d'attention pour cette variable
            resultat_attention = torch.zeros_like(requete)
            
            # Pour chaque variable (comme clé/valeur)
            for var_cle in noms_variables:
                # Extraire les caractéristiques de la variable clé/valeur
                cle_valeur = caracteristiques_variables[var_cle]  # [B, T, D]
                
                # Projeter la clé et la valeur
                cle_projetee = self.projection_cle(cle_valeur)  # [B, T, H*K]
                cle_projetee = cle_projetee.view(batch_size, seq_len, self.tetes, self.dim_cle)
                cle_projetee = cle_projetee.permute(0, 2, 1, 3)  # [B, H, T, K]
                
                valeur_projetee = self.projection_valeur(cle_valeur)  # [B, T, H*V]
                valeur_projetee = valeur_projetee.view(batch_size, seq_len, self.tetes, self.dim_valeur)
                valeur_projetee = valeur_projetee.permute(0, 2, 1, 3)  # [B, H, T, V]
                
                # Calculer les scores d'attention
                scores = torch.matmul(requete_projetee, cle_projetee.transpose(-2, -1))  # [B, H, T, T]
                scores = scores / (self.dim_cle ** 0.5)  # Mise à l'échelle
                
                # Appliquer softmax pour obtenir les poids d'attention
                poids_attention = F.softmax(scores, dim=-1)  # [B, H, T, T]
                poids_attention = self.dropout(poids_attention)
                
                # Appliquer l'attention aux valeurs
                contexte = torch.matmul(poids_attention, valeur_projetee)  # [B, H, T, V]
                contexte = contexte.permute(0, 2, 1, 3).contiguous()  # [B, T, H, V]
                contexte = contexte.view(batch_size, seq_len, self.tetes * self.dim_valeur)  # [B, T, H*V]
                
                # Projeter le contexte
                sortie = self.projection_sortie(contexte)  # [B, T, D]
                
                # Ajouter à l'attention cumulée
                resultat_attention += sortie
            
            # Normaliser par le nombre de variables
            resultat_attention = resultat_attention / len(noms_variables)
            
            # Connexion résiduelle et normalisation
            resultat_attention = self.norm(requete + resultat_attention)
            
            # Stocker le résultat
            caracteristiques_mises_a_jour[var_requete] = resultat_attention
        
        return caracteristiques_mises_a_jour


class AttentionTemporelle(nn.Module):
    """
    Module d'attention temporelle pour identifier les périodes critiques.
    """
    def __init__(self, dim_entree, config=None):
        """
        Initialise le module d'attention temporelle.
        
        Args:
            dim_entree (int): Dimension des caractéristiques d'entrée
            config (dict, optional): Configuration du module
        """
        super(AttentionTemporelle, self).__init__()
        config = config or CONFIG_ATTENTION['attention_temporelle']
        
        self.dim_cle = config['dim_cle']
        self.fenetres = config['fenetres']  # Tailles des fenêtres d'attention
        
        # Projections pour les clés et requêtes
        self.projection_cle = nn.Linear(dim_entree, self.dim_cle)
        self.projection_requete = nn.Linear(dim_entree, self.dim_cle)
        
        # Projection de sortie
        self.projection_sortie = nn.Linear(dim_entree, dim_entree)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(dim_entree)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Propagation avant du module d'attention temporelle.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, D]
            
        Returns:
            Tensor: Tensor de sortie avec attention temporelle
        """
        batch_size, seq_len, dim = x.size()
        
        # Projeter les clés et requêtes
        cles = self.projection_cle(x)  # [B, T, K]
        requetes = self.projection_requete(x)  # [B, T, K]
        
        # Initialiser le résultat d'attention
        resultat_attention = torch.zeros_like(x)
        
        # Pour chaque taille de fenêtre
        for fenetre in self.fenetres:
            # Créer un masque pour limiter l'attention à la fenêtre
            masque = torch.zeros(seq_len, seq_len, device=x.device)
            for i in range(seq_len):
                debut = max(0, i - fenetre // 2)
                fin = min(seq_len, i + fenetre // 2 + 1)
                masque[i, debut:fin] = 1.0
            
            # Calculer les scores d'attention
            scores = torch.bmm(requetes, cles.transpose(1, 2))  # [B, T, T]
            scores = scores / (self.dim_cle ** 0.5)  # Mise à l'échelle
            
            # Appliquer le masque
            scores = scores.masked_fill(masque.unsqueeze(0) == 0, -1e9)
            
            # Appliquer softmax pour obtenir les poids d'attention
            poids_attention = F.softmax(scores, dim=-1)  # [B, T, T]
            poids_attention = self.dropout(poids_attention)
            
            # Appliquer l'attention aux valeurs
            contexte = torch.bmm(poids_attention, x)  # [B, T, D]
            
            # Ajouter à l'attention cumulée
            resultat_attention += contexte
        
        # Normaliser par le nombre de fenêtres
        resultat_attention = resultat_attention / len(self.fenetres)
        
        # Projeter le résultat
        resultat_attention = self.projection_sortie(resultat_attention)
        
        # Connexion résiduelle et normalisation
        resultat_attention = self.norm(x + resultat_attention)
        
        return resultat_attention


class AttentionSpatiale(nn.Module):
    """
    Module d'attention spatiale pour les prédictions spécifiques aux coordonnées.
    """
    def __init__(self, canaux, config=None):
        """
        Initialise le module d'attention spatiale.
        
        Args:
            canaux (int): Nombre de canaux d'entrée
            config (dict, optional): Configuration du module
        """
        super(AttentionSpatiale, self).__init__()
        config = config or CONFIG_ATTENTION['attention_spatiale']
        
        self.canaux = canaux
        self.reduction = config['reduction']
        
        # Couches pour l'attention de canal (squeeze and excitation)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(canaux, canaux // self.reduction)
        self.fc2 = nn.Linear(canaux // self.reduction, canaux)
        
        # Couches pour l'attention spatiale
        self.conv_spatiale = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Propagation avant du module d'attention spatiale.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie avec attention spatiale
        """
        batch_size, canaux, hauteur, largeur = x.size()
        
        # Attention de canal (squeeze and excitation)
        y_canal = self.avg_pool(x).view(batch_size, canaux)
        y_canal = F.relu(self.fc1(y_canal))
        y_canal = torch.sigmoid(self.fc2(y_canal)).view(batch_size, canaux, 1, 1)
        
        # Appliquer l'attention de canal
        x_canal = x * y_canal
        
        # Attention spatiale
        avg_out = torch.mean(x_canal, dim=1, keepdim=True)
        max_out, _ = torch.max(x_canal, dim=1, keepdim=True)
        y_spatial = torch.cat([avg_out, max_out], dim=1)
        y_spatial = self.conv_spatiale(y_spatial)
        
        # Appliquer l'attention spatiale
        x_spatial = x_canal * y_spatial
        
        return x_spatial


class ModuleAttention(nn.Module):
    """
    Module combinant les différents mécanismes d'attention.
    """
    def __init__(self, dim_entree=256, canaux_spatiaux=128):
        """
        Initialise le module d'attention combiné.
        
        Args:
            dim_entree (int): Dimension des caractéristiques d'entrée
            canaux_spatiaux (int): Nombre de canaux pour l'attention spatiale
        """
        super(ModuleAttention, self).__init__()
        
        # Modules d'attention
        self.attention_croisee = AttentionCroiseeVariables(dim_entree)
        self.attention_temporelle = AttentionTemporelle(dim_entree)
        self.attention_spatiale = AttentionSpatiale(canaux_spatiaux)
    
    def forward_croisee(self, caracteristiques_variables):
        """
        Applique l'attention croisée entre variables.
        
        Args:
            caracteristiques_variables (dict): Dictionnaire des caractéristiques par variable
            
        Returns:
            dict: Caractéristiques mises à jour avec attention croisée
        """
        return self.attention_croisee(caracteristiques_variables)
    
    def forward_temporelle(self, x):
        """
        Applique l'attention temporelle.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, D]
            
        Returns:
            Tensor: Tensor avec attention temporelle
        """
        return self.attention_temporelle(x)
    
    def forward_spatiale(self, x):
        """
        Applique l'attention spatiale.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, C, H, W]
            
        Returns:
            Tensor: Tensor avec attention spatiale
        """
        return self.attention_spatiale(x)
    
    def forward(self, caracteristiques_variables, caracteristiques_spatiales=None):
        """
        Propagation avant du module d'attention combiné.
        
        Args:
            caracteristiques_variables (dict): Dictionnaire des caractéristiques par variable
            caracteristiques_spatiales (Tensor, optional): Caractéristiques spatiales
            
        Returns:
            tuple: (caractéristiques avec attention croisée et temporelle,
                   caractéristiques spatiales avec attention)
        """
        # Appliquer l'attention croisée entre variables
        caracteristiques_croisees = self.forward_croisee(caracteristiques_variables)
        
        # Appliquer l'attention temporelle à chaque variable
        caracteristiques_temporelles = {}
        for var, feat in caracteristiques_croisees.items():
            caracteristiques_temporelles[var] = self.forward_temporelle(feat)
        
        # Appliquer l'attention spatiale si des caractéristiques spatiales sont fournies
        if caracteristiques_spatiales is not None:
            caracteristiques_spatiales = self.forward_spatiale(caracteristiques_spatiales)
        
        return caracteristiques_temporelles, caracteristiques_spatiales


if __name__ == "__main__":
    # Test des mécanismes d'attention
    print("Test des mécanismes d'attention...")
    
    # Créer des données de test
    batch_size = 2
    seq_len = 10
    dim_features = 256
    canaux_spatiaux = 128
    hauteur = 14
    largeur = 14
    
    # Test de l'attention croisée
    print("\nTest de l'attention croisée:")
    caracteristiques_variables = {
        'var1': torch.randn(batch_size, seq_len, dim_features),
        'var2': torch.randn(batch_size, seq_len, dim_features),
        'var3': torch.randn(batch_size, seq_len, dim_features)
    }
    
    attention_croisee = AttentionCroiseeVariables(dim_features)
    caracteristiques_croisees = attention_croisee(caracteristiques_variables)
    
    for var, feat in caracteristiques_croisees.items():
        print(f"  {var}: {feat.shape}")
    
    # Test de l'attention temporelle
    print("\nTest de l'attention temporelle:")
    x_temporel = torch.randn(batch_size, seq_len, dim_features)
    
    attention_temporelle = AttentionTemporelle(dim_features)
    x_temporel_attention = attention_temporelle(x_temporel)
    
    print(f"  Forme d'entrée: {x_temporel.shape}")
    print(f"  Forme de sortie: {x_temporel_attention.shape}")
    
    # Test de l'attention spatiale
    print("\nTest de l'attention spatiale:")
    x_spatial = torch.randn(batch_size, canaux_spatiaux, hauteur, largeur)
    
    attention_spatiale = AttentionSpatiale(canaux_spatiaux)
    x_spatial_attention = attention_spatiale(x_spatial)
    
    print(f"  Forme d'entrée: {x_spatial.shape}")
    print(f"  Forme de sortie: {x_spatial_attention.shape}")
    
    # Test du module d'attention combiné
    print("\nTest du module d'attention combiné:")
    module_attention = ModuleAttention(dim_features, canaux_spatiaux)
    
    caracteristiques_temporelles, caracteristiques_spatiales = module_attention(
        caracteristiques_variables, x_spatial)
    
    print("  Caractéristiques temporelles:")
    for var, feat in caracteristiques_temporelles.items():
        print(f"    {var}: {feat.shape}")
    
    print(f"  Caractéristiques spatiales: {caracteristiques_spatiales.shape}")