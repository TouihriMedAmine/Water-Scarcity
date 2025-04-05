# -*- coding: utf-8 -*-
"""
Module de fusion temporelle pour le modèle de Deep Learning d'optimisation d'irrigation

Ce module implémente les mécanismes de fusion temporelle qui combinent les caractéristiques
extraites par les encodeurs spécifiques aux variables à travers le temps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

# Ajouter le répertoire parent au chemin pour pouvoir importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_FUSION_TEMPORELLE


class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM pour le traitement de séquences d'images.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialise une cellule ConvLSTM.
        
        Args:
            input_dim (int): Nombre de canaux d'entrée
            hidden_dim (int): Nombre de canaux cachés
            kernel_size (tuple): Taille du noyau de convolution
            bias (bool): Utiliser un biais ou non
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # Convolution pour les portes d'entrée, de sortie, d'oubli et la cellule
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, o, g
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        """
        Propagation avant de la cellule ConvLSTM.
        
        Args:
            input_tensor (Tensor): Tensor d'entrée de forme [B, C, H, W]
            cur_state (tuple): État courant (h, c)
            
        Returns:
            tuple: Nouvel état (h, c)
        """
        h_cur, c_cur = cur_state
        
        # Concaténer l'entrée et l'état caché
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Calculer les portes
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Appliquer les fonctions d'activation
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Mettre à jour l'état de la cellule et l'état caché
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """
        Initialise l'état caché et l'état de la cellule.
        
        Args:
            batch_size (int): Taille du batch
            image_size (tuple): Taille de l'image (hauteur, largeur)
            
        Returns:
            tuple: État initial (h, c)
        """
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """
    Module ConvLSTM pour le traitement de séquences d'images.
    """
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, batch_first=True, dropout=0.0):
        """
        Initialise un module ConvLSTM.
        
        Args:
            input_dim (int): Nombre de canaux d'entrée
            hidden_dims (list): Liste des dimensions cachées pour chaque couche
            kernel_size (tuple): Taille du noyau de convolution
            num_layers (int): Nombre de couches
            batch_first (bool): Si True, l'entrée est de forme [B, T, C, H, W]
            dropout (float): Taux de dropout
        """
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Créer les cellules ConvLSTM
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dims[i], self.kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
        
        # Couche de dropout
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Propagation avant du module ConvLSTM.
        
        Args:
            input_tensor (Tensor): Tensor d'entrée de forme [B, T, C, H, W] si batch_first=True
            hidden_state (list): État caché initial
            
        Returns:
            tuple: (sortie, état caché final)
        """
        # Réorganiser l'entrée si nécessaire
        if self.batch_first:
            # [B, T, C, H, W] -> [T, B, C, H, W]
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        # Dimensions
        b, _, _, h, w = input_tensor.size()
        
        # Générer l'état caché initial si non fourni
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        # Séquence de sortie pour chaque couche
        layer_output_list = []
        last_state_list = []
        
        # Séquence d'entrée
        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor
        
        # Pour chaque couche
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Pour chaque pas de temps
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[t], (h, c))
                output_inner.append(h)
            
            # Appliquer le dropout sauf pour la dernière couche
            layer_output = torch.stack(output_inner, dim=0)
            if layer_idx != self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            # Préparer l'entrée pour la couche suivante
            cur_layer_input = layer_output
            
            # Stocker la sortie et l'état final
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        # Réorganiser la sortie si nécessaire
        if self.batch_first:
            # [T, B, C, H, W] -> [B, T, C, H, W]
            layer_output = layer_output.permute(1, 0, 2, 3, 4)
        
        return layer_output, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        """
        Initialise l'état caché pour toutes les couches.
        
        Args:
            batch_size (int): Taille du batch
            image_size (tuple): Taille de l'image (hauteur, largeur)
            
        Returns:
            list: Liste des états cachés initiaux pour chaque couche
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel pour le Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initialise l'encodage positionnel.
        
        Args:
            d_model (int): Dimension du modèle
            max_len (int): Longueur maximale de la séquence
        """
        super(PositionalEncoding, self).__init__()
        
        # Créer une matrice de même forme que l'embedding d'entrée
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculer l'encodage positionnel
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Ajouter une dimension batch
        pe = pe.unsqueeze(0)
        
        # Enregistrer l'encodage positionnel comme buffer (non comme paramètre)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Ajoute l'encodage positionnel à l'entrée.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, D]
            
        Returns:
            Tensor: Tensor avec encodage positionnel
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerTemporel(nn.Module):
    """
    Module Transformer pour le traitement de séquences temporelles.
    """
    def __init__(self, config=None):
        """
        Initialise un module Transformer temporel.
        
        Args:
            config (dict, optional): Configuration du Transformer
        """
        super(TransformerTemporel, self).__init__()
        config = config or CONFIG_FUSION_TEMPORELLE['type_transformer']
        
        self.dim_modele = config['dim_modele']
        self.tetes_attention = config['tetes_attention']
        self.profondeur = config['profondeur']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        
        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(self.dim_modele)
        
        # Couche d'encodeur Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_modele,
            nhead=self.tetes_attention,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.profondeur)
    
    def forward(self, x, mask=None):
        """
        Propagation avant du module Transformer temporel.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, D]
            mask (Tensor, optional): Masque d'attention
            
        Returns:
            Tensor: Tensor de sortie
        """
        # Appliquer l'encodage positionnel
        x = self.pos_encoder(x)
        
        # Passer à travers l'encodeur Transformer
        output = self.transformer_encoder(x, mask=mask)
        
        return output


class ModuleFusionTemporelle(nn.Module):
    """
    Module de fusion temporelle combinant ConvLSTM et Transformer.
    """
    def __init__(self, dim_entree=256, config=None):
        """
        Initialise le module de fusion temporelle.
        
        Args:
            dim_entree (int): Dimension d'entrée
            config (dict, optional): Configuration du module
        """
        super(ModuleFusionTemporelle, self).__init__()
        self.dim_entree = dim_entree
        
        # Configurations
        config_convlstm = CONFIG_FUSION_TEMPORELLE['type_convlstm']
        config_transformer = CONFIG_FUSION_TEMPORELLE['type_transformer']
        
        # Paramètres ConvLSTM
        filtres_convlstm = config_convlstm['filtres']
        taille_noyau_convlstm = config_convlstm['taille_noyau']
        dropout_convlstm = config_convlstm['dropout']
        
        # Paramètres Transformer
        dim_modele_transformer = config_transformer['dim_modele']
        
        # Projection pour adapter la dimension d'entrée au ConvLSTM
        self.projection_convlstm = nn.Conv2d(dim_entree, filtres_convlstm[0], kernel_size=1)
        
        # Module ConvLSTM pour les motifs à court terme
        self.convlstm = ConvLSTM(
            input_dim=filtres_convlstm[0],
            hidden_dims=filtres_convlstm,
            kernel_size=taille_noyau_convlstm,
            num_layers=len(filtres_convlstm),
            batch_first=True,
            dropout=dropout_convlstm
        )
        
        # Projection pour adapter la sortie du ConvLSTM au Transformer
        self.projection_transformer = nn.Linear(filtres_convlstm[-1], dim_modele_transformer)
        
        # Module Transformer pour les dépendances à long terme
        self.transformer = TransformerTemporel(config_transformer)
        
        # Couche de sortie
        self.fc_sortie = nn.Linear(dim_modele_transformer, dim_entree)
    
    def forward(self, x):
        """
        Propagation avant du module de fusion temporelle.
        
        Args:
            x (Tensor): Tensor d'entrée de forme [B, T, D] ou [B, T, C, H, W]
            
        Returns:
            Tensor: Tensor de sortie
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Vérifier si l'entrée est spatiale (5D) ou déjà aplatie (3D)
        if len(x.shape) == 5:  # [B, T, C, H, W]
            # Traitement ConvLSTM pour les caractéristiques spatiales
            x_spatial = x
            
            # Projeter les caractéristiques pour le ConvLSTM
            x_projected = []
            for t in range(seq_len):
                x_t = self.projection_convlstm(x_spatial[:, t])
                x_projected.append(x_t)
            x_projected = torch.stack(x_projected, dim=1)  # [B, T, C', H, W]
            
            # Appliquer le ConvLSTM
            convlstm_out, _ = self.convlstm(x_projected)
            
            # Extraire les caractéristiques globales
            convlstm_features = F.adaptive_avg_pool2d(convlstm_out[:, -1], 1).view(batch_size, -1)
            
            # Projeter pour le Transformer
            transformer_in = self.projection_transformer(convlstm_features).unsqueeze(1)  # [B, 1, D]
            
        else:  # [B, T, D]
            # Entrée déjà aplatie, passer directement au Transformer
            transformer_in = x
        
        # Appliquer le Transformer pour les dépendances à long terme
        transformer_out = self.transformer(transformer_in)
        
        # Couche de sortie
        output = self.fc_sortie(transformer_out)
        
        return output


if __name__ == "__main__":
    # Test du module de fusion temporelle
    print("Test du module de fusion temporelle...")
    
    # Créer des données de test
    batch_size = 2
    seq_len = 10
    dim_features = 256
    
    # Test avec entrée aplatie
    x_flat = torch.randn(batch_size, seq_len, dim_features)
    print(f"Forme de l'entrée aplatie: {x_flat.shape}")
    
    fusion_module = ModuleFusionTemporelle(dim_entree=dim_features)
    output_flat = fusion_module(x_flat)
    print(f"Forme de la sortie: {output_flat.shape}")
    
    # Test avec entrée spatiale
    channels = 64
    height = 7
    width = 7
    x_spatial = torch.randn(batch_size, seq_len, channels, height, width)
    print(f"\nForme de l'entrée spatiale: {x_spatial.shape}")
    
    fusion_module_spatial = ModuleFusionTemporelle(dim_entree=channels)
    output_spatial = fusion_module_spatial(x_spatial)
    print(f"Forme de la sortie: {output_spatial.shape}")