Prédiction du Ruissellement avec ConvLSTM

1. Description

Ce projet utilise un modèle ConvLSTM pour prédire le ruissellement (débit de surface) sur la région CONUS des États-Unis. Le modèle apprend à partir de séquences temporelles de données hydrométéorologiques et produit une carte de débit à un horizon de 6 jours.

2. Jeu de données

Les données, fournies par la NASA (datanasa), sont stockées sous forme d’images PNG en niveaux de gris dans quatre dossiers :

Rainf : précipitations quotidiennes

SoilM_0_10cm : humidité du sol (0–10 cm)

Qs : débit de surface total

Qsb : débit de base

Chaque image est nommée YYYYMMDD.png et normalisée entre 0 et 1.

3. Entrées / Sorties

Entrées : séquence de T = 6 pas de temps, chaque pas représenté par 4 canaux (précip., sol, Qs, Qsb).Format : (batch_size, T, 4, H, W)

Sortie : carte de débit de surface à t + Δ (Δ = 6), format (batch_size, 1, H, W).

4. Architecture et fonctionnement

Le modèle est un encoder–decoder ConvLSTM :

L’encodeur traite les 6 pas historiques pour extraire des caractéristiques spatio-temporelles.

Le décodeur génère la prédiction du débit de surface à l’horizon.

Entraînement :

Critère : MSE (Mean Squared Error)

Optimiseur : Adam (lr=1e-4)

Scheduler : ReduceLROnPlateau

Précision mixte (torch.cuda.amp)

Early stopping (patience = 8)

5. Interprétation des courbes d’apprentissage



Baisse rapide des pertes en début d’entraînement.

Plateau autour de 0.001 après ~10 époques.

Courbes train/val proches → bon compromis biais/variance.