Prédiction du ruissellement de surface (QS) avec ConvLSTM

Objectif

Ce projet a pour objectif de prédire la carte de ruissellement de surface (QS) à l'échelle spatiale pour un jour futur, en s'appuyant sur une séquence de cartes historiques.

Qu'est-ce qu'on prédit ?

Le modèle ConvLSTM génère une estimation de la distribution spatiale du ruissellement de surface (QS) au jour suivant (ou à T+6 jours selon la configuration choisie).

Entrées (Input)

Séquences de N images QS (par défaut 6 jours consécutifs) au format TIFF/PNG, stockées dans data/input/.

Chaque échantillon a la forme : (batch_size, sequence_length, channels, height, width) où channels=1 pour QS.

Sorties (Output)

Carte QS prédite pour le jour suivant, sauvegardée dans outputs/predictions/.

Courbes d'entraînement et de validation (loss) générées dans outputs/plots/.

Structure du dépôt

data_loader.py : Chargement, normalisation et création de batches temporels de données QS.

model.py : Définition de la ConvLSTMCell et du modèle ConvLSTMForecaster.

train.py : Boucle d'entraînement, calcul de la loss (MSE), sauvegarde des checkpoints.

infer.py : Script pour lancer l'inférence sur des données de test et générer les visualisations.

utils.py : Fonctions utilitaires (affichage des images, calcul de métriques, gestion des chemins).

requirements.txt / environment.yml : Liste des dépendances Python.

Installation

Cloner le dépôt :

git clone <url_du_repository>
cd <nom_du_repository>

Créer un environnement virtuel et installer les dépendances :

pip install -r requirements.txt
# ou avec conda :
# conda env create -f environment.yml && conda activate runoff

Utilisation

Entraînement

Lancer l'entraînement du modèle :

python train.py \
  --data_dir data/input \
  --epochs 50 \
  --batch_size 8 \
  --hidden_channels 16 \
  --kernel_size 3

Inférence

Générer des prédictions sur un jeu de test :

python infer.py \
  --weights_path checkpoints/model_epoch50.pth \
  --input_dir data/test \
  --output_dir outputs/predictions

Explication du fonctionnement

Chargement des données :

Les images QS sont lues, normalisées et regroupées en séquences temporelles.

Modèle ConvLSTM :

La ConvLSTMCell combine une convolution 2D et les mécanismes LSTM pour capturer les dynamiques spatio-temporelles.

Le ConvLSTMForecaster enchaîne plusieurs cellules pour traiter la séquence complète et produire la prédiction finale.

Entraînement :

Optimisation de la Mean Squared Error (MSE) entre la carte prédite et la carte réelle.

Possibilité d'utiliser un scheduler de taux d'apprentissage et un early stopping pour éviter le surapprentissage.

Inférence :

Chargement du checkpoint du modèle.

Propagation avant pour générer la carte QS prédite.

Sauvegarde des résultats et visualisation comparative (prédiction vs vérité terrain).

Résultats et évaluation

Visualisation des courbes d'entraînement et de validation pour suivre la convergence.

Affichage côte-à-côte des cartes QS réelles et prédites pour évaluer qualitativement la performance.

Ce README offre une vue d'ensemble pour comprendre, exécuter et étendre le code de prédiction du ruissellement de surface à l'aide d'un modèle ConvLSTM.

