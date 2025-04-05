# Module de Deep Learning pour l'Optimisation d'Irrigation

Ce module implémente l'architecture de Deep Learning décrite dans le README principal pour l'optimisation de l'irrigation agricole. Il utilise des techniques avancées de traitement d'images satellitaires et d'apprentissage profond pour prédire les besoins en irrigation et fournir des recommandations adaptées.

## Structure du Module

Le module est organisé en plusieurs fichiers Python, chacun responsable d'une partie spécifique du système :

- `config.py` : Configuration centralisée pour tous les paramètres du modèle
- `data_pipeline.py` : Gestion du chargement et du prétraitement des données
- `encodeurs.py` : Implémentation des encodeurs spécifiques aux variables
- `fusion_temporelle.py` : Module de fusion des caractéristiques temporelles
- `mecanismes_attention.py` : Mécanismes d'attention pour améliorer les prédictions
- `systeme_recommandation.py` : Système de recommandation basé sur les prédictions
- `modele_principal.py` : Intégration de tous les composants en un modèle complet
- `main.py` : Point d'entrée principal pour l'entraînement et l'inférence

## Architecture du Modèle

L'architecture du modèle suit la méthodologie décrite dans le README principal :

1. **Pipeline de Données** : Structure temporelle multi-variable avec indexation géospatiale
2. **Encodeurs Spécifiques aux Variables** :
   - CNN 3D avec connexions résiduelles pour ESoil_tavg
   - U-Net avec connexions skip pour RootMoist_inst
   - Vision Transformer (ViT) pour Tair_f_inst
   - Temporal ConvNet pour Rainf_tavg
3. **Module de Fusion Temporelle** : Combinaison de ConvLSTM et Transformer
4. **Mécanismes d'Attention** : Attention croisée entre variables, temporelle et spatiale
5. **Système de Recommandation** : Analyse du sol, recommandations de cultures et stratégies d'irrigation

## Utilisation

### Installation des Dépendances

```bash
pip install torch torchvision numpy pandas matplotlib pillow h5py cartopy scikit-learn
```

### Entraînement du Modèle

```bash
python main.py entrainer --repertoire-images ../visualization_output --taille-sequence 30 --taille-lot 32 --epoques 100
```

### Évaluation du Modèle

```bash
python main.py evaluer --repertoire-images ../visualization_output --chemin-modele ./modeles/meilleur_modele.pth
```

### Inférence et Recommandations

```bash
python main.py inferer --date 2023-05-15 --latitude 34.05 --longitude -118.25 --repertoire-images ../visualization_output
```

Pour spécifier l'utilisation actuelle des terres et les contraintes d'eau :

```bash
python main.py inferer --date 2023-05-15 --latitude 34.05 --longitude -118.25 --utilisation-actuelle '{"culture": "maïs", "superficie": 10.0}' --contraintes-eau '{"disponibilite": "limitée", "source": "puits"}'
```

## Apprentissage Multi-Tâche

Le modèle est entraîné avec une approche d'apprentissage multi-tâche :

1. **Tâche Principale** : Prédiction de la quantité d'irrigation optimale
2. **Tâches Auxiliaires** :
   - Prévision de l'humidité du sol
   - Estimation de l'évapotranspiration
   - Classification du stress hydrique
   - Classification du type de sol
   - Prédiction de l'adéquation des cultures

La fonction de perte combinée est pondérée selon l'importance de chaque tâche :

```
L = 0.5*MSE_irrigation + 0.2*MSE_humidite + 0.1*MSE_evapotranspiration + 0.1*MSE_stress + 0.05*CE_type_sol + 0.05*MSE_adequation
```

## Système de Recommandation

Le système de recommandation fournit trois types d'informations :

1. **Analyse du Sol** : Type de sol, capacité de rétention d'eau, teneur en nutriments
2. **Recommandations de Cultures** : Top 3 des cultures adaptées avec rendements estimés
3. **Stratégie d'Irrigation** : Méthode optimale, quantité d'eau, fréquence et durée

Les recommandations sont générées en fonction des coordonnées géographiques, de la date, et des prédictions du modèle.