# -*- coding: utf-8 -*-
"""
Encodeurs spécifiques aux variables pour le modèle de Deep Learning d'optimisation d'irrigation

Ce module implémente différents encodeurs pour traiter les différentes variables
environnementales utilisées dans le modèle d'optimisation d'irrigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ajouter le répertoire parent au chemin pour pouvoir importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_ENCODEURS


class BlocResiduel(nn.Module):
    """
    Bloc résiduel pour les réseaux CNN avec connexions résiduelles.
    """
    def __init__(self, canaux_entree, canaux_sortie, stride=1):
        """
        Initialise un bloc résiduel.
        
        Args:
            canaux_entree (int): Nombre de canaux d'entrée
            canaux_sortie (int): Nombre de canaux de sortie
            stride (int): Pas de convolution
        """
        super(BlocResiduel, self).__init__()
        self.conv1 = nn.Conv2d(canaux_entree, canaux_sortie, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canaux_sortie)
        self.conv2 = nn.Conv2d(canaux_sortie, canaux_sortie, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canaux_sortie)
        
        # Connexion résiduelle
        self.shortcut = nn.Sequential()
        if stride != 1 or canaux_entree != canaux_sortie:
            self.shortcut = nn.Sequential(
                nn.Conv2d(canaux_entree, canaux_sortie, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(canaux_sortie)
            )
    
    def forward(self, x):
        """
        Propagation avant du bloc résiduel.
        
        Args:
            x (Tensor): Tensor d'entrée
            
        Returns:
            Tensor: Tensor de sortie
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN3D(nn.Module):
    """
    Encodeur 3D CNN avec connexions résiduelles pour ESoil_tavg.
    Capture les gradients de chaleur du sol.
    """
    def __init__(self, config=None):
        """
        Initialise l'encodeur 3D CNN.
        
        Args:
            config (dict, optional): Configuration de l'encodeur
        """
        super(CNN3D, self).__init__()
        config = config or CONFIG_ENCODEURS['ESoil_tavg']
        filtres = config['filtres']
        taille_noyau = config['taille_noyau']
        connexions_residuelles = config.get('connexions_residuelles', True)
        
        # Couche d'entrée
        self.conv1 = nn.Conv3d(3, filtres[0], kernel_size=taille_noyau, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(filtres[0])
        
        # Couches cachées avec connexions résiduelles
        self.layers = nn.ModuleList()
        for i in range(len(filtres) - 1):
            if connexions_residuelles:
                self.layers.append(self._make_residual_layer_3d(filtres[i], filtres[i+1]))
            else:
                self.layers.append(self._make_conv_layer_3d(filtres[i], filtres[i+1]))
        
        # Couche de sortie
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(filtres[-1], 256)
    
    def _make_conv_layer_3d(self, canaux_entree, canaux_sortie):
        """
        Crée une couche de convolution 3D simple.
        
        Args:
            canaux_entree (int): Nombre de canaux d'entrée
            canaux_sortie (int): Nombre de canaux de sortie
            
        Returns:
            nn.Sequential: Couche de convolution 3D
        """
        return nn.Sequential(
            nn.Conv3d(canaux_entree, canaux_sortie, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(canaux_sortie),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    
    def _make_residual_layer_3d(self, canaux_entree, canaux_sortie):
        """
        Crée une couche résiduelle 3D.
        
        Args:
            canaux_entree (int): Nombre de canaux d'entrée
            canaux_sortie (int): Nombre de canaux de sortie
            
        Returns:
            nn.Sequential: Couche résiduelle 3D
        """
        return nn.Sequential(
            nn.Conv3d(canaux_entree, canaux_sortie, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(canaux_sortie),
            nn.ReLU(inplace=True),
            nn.Conv3d(canaux_sortie, canaux_sortie, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(canaux_sortie),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        """
        Propagation avant de l'encodeur 3D CNN.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        # x est de forme [B, T, C, H, W], le réorganiser en [B, C, T, H, W] pour Conv3D
        x = x.permute(0, 2, 1, 3, 4)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class UNet(nn.Module):
    """
    Encodeur U-Net avec connexions skip pour RootMoist_inst.
    Permet une segmentation précise de la zone racinaire.
    """
    def __init__(self, config=None):
        """
        Initialise l'encodeur U-Net.
        
        Args:
            config (dict, optional): Configuration de l'encodeur
        """
        super(UNet, self).__init__()
        config = config or CONFIG_ENCODEURS['RootMoist_inst']
        filtres = config['filtres']
        connexions_skip = config.get('connexions_skip', True)
        self.connexions_skip = connexions_skip
        
        # Encodeur
        self.enc_conv1 = self._double_conv(3, filtres[0])
        self.enc_conv2 = self._double_conv(filtres[0], filtres[1])
        self.enc_conv3 = self._double_conv(filtres[1], filtres[2])
        self.enc_conv4 = self._double_conv(filtres[2], filtres[3])
        
        self.pool = nn.MaxPool2d(2)
        
        # Décodeur
        self.upconv3 = nn.ConvTranspose2d(filtres[3], filtres[2], kernel_size=2, stride=2)
        self.dec_conv3 = self._double_conv(filtres[3], filtres[2])
        
        self.upconv2 = nn.ConvTranspose2d(filtres[2], filtres[1], kernel_size=2, stride=2)
        self.dec_conv2 = self._double_conv(filtres[2], filtres[1])
        
        self.upconv1 = nn.ConvTranspose2d(filtres[1], filtres[0], kernel_size=2, stride=2)
        self.dec_conv1 = self._double_conv(filtres[1], filtres[0])
        
        # Couche de sortie
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filtres[0], 256)
    
    def _double_conv(self, in_channels, out_channels):
        """
        Crée un bloc de double convolution pour U-Net.
        
        Args:
            in_channels (int): Nombre de canaux d'entrée
            out_channels (int): Nombre de canaux de sortie
            
        Returns:
            nn.Sequential: Bloc de double convolution
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Propagation avant de l'encodeur U-Net.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        # Traiter chaque image de la séquence indépendamment
        batch_size, seq_len, channels, height, width = x.size()
        features = []
        
        for t in range(seq_len):
            # Extraire l'image à l'instant t
            x_t = x[:, t, :, :, :]
            
            # Encodeur
            enc1 = self.enc_conv1(x_t)
            enc2 = self.enc_conv2(self.pool(enc1))
            enc3 = self.enc_conv3(self.pool(enc2))
            enc4 = self.enc_conv4(self.pool(enc3))
            
            # Décodeur avec connexions skip
            dec3 = self.upconv3(enc4)
            if self.connexions_skip:
                dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec_conv3(dec3)
            
            dec2 = self.upconv2(dec3)
            if self.connexions_skip:
                dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec_conv2(dec2)
            
            dec1 = self.upconv1(dec2)
            if self.connexions_skip:
                dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec_conv1(dec1)
            
            # Extraction des caractéristiques
            feat = self.global_avg_pool(dec1)
            feat = feat.view(batch_size, -1)
            feat = self.fc(feat)
            
            features.append(feat)
        
        # Combiner les caractéristiques de tous les pas de temps
        features = torch.stack(features, dim=1)  # [B, T, 256]
        
        return features


class VisionTransformer(nn.Module):
    """
    Encodeur Vision Transformer (ViT) pour Tair_f_inst.
    Capture les motifs thermiques globaux.
    """
    def __init__(self, config=None):
        """
        Initialise l'encodeur Vision Transformer.
        
        Args:
            config (dict, optional): Configuration de l'encodeur
        """
        super(VisionTransformer, self).__init__()
        config = config or CONFIG_ENCODEURS['Tair_f_inst']
        taille_patch = config['taille_patch']
        dim_embedding = config['dim_embedding']
        profondeur = config['profondeur']
        tetes_attention = config['tetes_attention']
        
        # Paramètres d'image
        self.taille_patch = taille_patch
        self.dim_embedding = dim_embedding
        
        # Projection des patches
        self.projection_patch = nn.Conv2d(3, dim_embedding, kernel_size=taille_patch, stride=taille_patch)
        
        # Token de classe [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_embedding))
        
        # Embeddings de position
        self.pos_embedding = nn.Parameter(torch.zeros(1, 197, dim_embedding))  # 196 patches + 1 cls token pour une image 224x224
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=tetes_attention,
            dim_feedforward=dim_embedding * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=profondeur)
        
        # Couche de sortie
        self.norm = nn.LayerNorm(dim_embedding)
        self.fc = nn.Linear(dim_embedding, 256)
    
    def forward(self, x):
        """
        Propagation avant de l'encodeur Vision Transformer.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        batch_size, seq_len, channels, height, width = x.size()
        features = []
        
        for t in range(seq_len):
            # Extraire l'image à l'instant t
            x_t = x[:, t, :, :, :]
            
            # Diviser l'image en patches et les projeter
            patches = self.projection_patch(x_t)  # [B, D, H/P, W/P]
            patches = patches.flatten(2).transpose(1, 2)  # [B, N, D]
            
            # Ajouter le token de classe [CLS]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x_t = torch.cat([cls_tokens, patches], dim=1)
            
            # Ajouter les embeddings de position
            x_t = x_t + self.pos_embedding[:, :x_t.size(1), :]
            
            # Passer à travers le Transformer Encoder
            x_t = self.transformer_encoder(x_t)
            
            # Utiliser le token [CLS] pour la classification
            x_t = x_t[:, 0]  # Token [CLS]
            x_t = self.norm(x_t)
            x_t = self.fc(x_t)
            
            features.append(x_t)
        
        # Combiner les caractéristiques de tous les pas de temps
        features = torch.stack(features, dim=1)  # [B, T, 256]
        
        return features


class TemporalConvNet(nn.Module):
    """
    Encodeur Temporal ConvNet pour Rainf_tavg.
    Détecte les événements de précipitation.
    """
    def __init__(self, config=None):
        """
        Initialise l'encodeur Temporal ConvNet.
        
        Args:
            config (dict, optional): Configuration de l'encodeur
        """
        super(TemporalConvNet, self).__init__()
        config = config or CONFIG_ENCODEURS['Rainf_tavg']
        filtres = config['filtres']
        taille_noyau = config['taille_noyau']
        dilatation = config['dilatation']
        
        # Couche de convolution spatiale pour extraire les caractéristiques de chaque image
        self.conv_spatiale = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Couches de convolution temporelle avec dilatation
        self.conv_temporelles = nn.ModuleList()
        in_channels = 64
        for i, (f, d) in enumerate(zip(filtres, dilatation)):
            self.conv_temporelles.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, f, kernel_size=taille_noyau, padding=taille_noyau//2*d, dilation=d),
                    nn.BatchNorm1d(f),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = f
        
        # Couche de sortie
        self.fc = nn.Linear(filtres[-1], 256)
    
    def forward(self, x):
        """
        Propagation avant de l'encodeur Temporal ConvNet.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Extraire les caractéristiques spatiales de chaque image
        spatial_features = []
        for t in range(seq_len):
            feat = self.conv_spatiale(x[:, t])
            spatial_features.append(feat.view(batch_size, -1))
        
        # Combiner les caractéristiques spatiales en une séquence temporelle
        spatial_features = torch.stack(spatial_features, dim=2)  # [B, C, T]
        
        # Appliquer les convolutions temporelles
        temporal_features = spatial_features
        for conv in self.conv_temporelles:
            temporal_features = conv(temporal_features)
        
        # Réorganiser pour obtenir des caractéristiques par pas de temps
        temporal_features = temporal_features.permute(0, 2, 1)  # [B, T, C]
        
        # Appliquer la couche fully connected à chaque pas de temps
        output_features = self.fc(temporal_features)  # [B, T, 256]
        
        return output_features


class EncodeurCNN(nn.Module):
    """
    Encodeur CNN standard pour les autres variables.
    """
    def __init__(self, config=None):
        """
        Initialise l'encodeur CNN standard.
        
        Args:
            config (dict, optional): Configuration de l'encodeur
        """
        super(EncodeurCNN, self).__init__()
        config = config or CONFIG_ENCODEURS['default']
        filtres = config['filtres']
        taille_noyau = config['taille_noyau']
        
        # Couches de convolution
        layers = []
        in_channels = 3
        for f in filtres:
            layers.extend([
                nn.Conv2d(in_channels, f, kernel_size=taille_noyau, padding=taille_noyau//2),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = f
        
        self.features = nn.Sequential(*layers)
        
        # Couche de sortie
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filtres[-1], 256)
    
    def forward(self, x):
        """
        Propagation avant de l'encodeur CNN standard.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        batch_size, seq_len, channels, height, width = x.size()
        features = []
        
        for t in range(seq_len):
            # Extraire l'image à l'instant t
            x_t = x[:, t, :, :, :]
            
            # Extraire les caractéristiques
            feat = self.features(x_t)
            feat = self.global_avg_pool(feat)
            feat = feat.view(batch_size, -1)
            feat = self.fc(feat)
            
            features.append(feat)
        
        # Combiner les caractéristiques de tous les pas de temps
        features = torch.stack(features, dim=1)  # [B, T, 256]
        
        return features


def get_encodeur(variable):
    """
    Retourne l'encodeur approprié pour la variable spécifiée.
    
    Args:
        variable (str): Nom de la variable
        
    Returns:
        nn.Module: Encodeur pour la variable
    """
    if variable == 'ESoil_tavg':
        return CNN3D()
    elif variable == 'RootMoist_inst':
        return UNet()
    elif variable == 'Tair_f_inst':
        return VisionTransformer()
    elif variable == 'Rainf_tavg':
        return TemporalConvNet()
    else:
        return EncodeurCNN()


if __name__ == "__main__":
    # Test des encodeurs
    print("Test des encodeurs...")
    
    # Créer des données de test
    batch_size = 2
    seq_len = 10
    channels = 3
    height = 224
    width = 224
    
    x = torch.randn(batch_size, seq_len, channels, height, width)
    
    # Tester chaque encodeur
    for var in ['ESoil_tavg', 'RootMoist_inst', 'Tair_f_inst', 'Rainf_tavg', 'SoilMoist_S_tavg']:
        print(f"\nTest de l'encodeur pour {var}")
        encodeur = get_encodeur(var)
        output = encodeur(x)
        print(f"Forme de sortie: {output.shape}")