# -*- coding: utf-8 -*-
"""
Point d'entrée principal pour le modèle de Deep Learning d'optimisation d'irrigation

Ce script permet d'exécuter l'entraînement, l'évaluation ou l'inférence du modèle
d'optimisation d'irrigation en fonction des arguments fournis.
"""

import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CHEMIN_DONNEES
from deep_learning.data_pipeline import creer_dataloaders
from deep_learning.modele_principal import ModeleIrrigationOptimisation, entrainer_modele, evaluer_modele


def creer_dossiers_necessaires():
    """
    Crée les dossiers nécessaires pour le stockage des modèles et des résultats.
    """
    os.makedirs(CHEMIN_DONNEES['modeles'], exist_ok=True)
    os.makedirs(CHEMIN_DONNEES['resultats'], exist_ok=True)


def entrainer(args):
    """
    Entraîne le modèle d'optimisation d'irrigation.
    
    Args:
        args (Namespace): Arguments de ligne de commande
    """
    print("Démarrage de l'entraînement du modèle...")
    
    # Créer les dataloaders
    train_loader, val_loader, test_loader = creer_dataloaders(
        repertoire_images=args.repertoire_images,
        taille_sequence=args.taille_sequence,
        taille_lot=args.taille_lot
    )
    
    # Créer le modèle
    modele = ModeleIrrigationOptimisation()
    
    # Utiliser le GPU si disponible
    if torch.cuda.is_available():
        modele = modele.cuda()
        print("Utilisation du GPU pour l'entraînement.")
    
    # Entraîner le modèle
    modele, historique = entrainer_modele(
        modele=modele,
        train_loader=train_loader,
        val_loader=val_loader,
        epoques=args.epoques,
        patience=args.patience,
        taux_apprentissage=args.taux_apprentissage
    )
    
    # Sauvegarder le modèle final
    torch.save(modele.state_dict(), os.path.join(CHEMIN_DONNEES['modeles'], 'modele_final.pth'))
    print(f"Modèle sauvegardé dans {os.path.join(CHEMIN_DONNEES['modeles'], 'modele_final.pth')}")
    
    # Tracer et sauvegarder la courbe d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.plot(historique['train'], label='Entraînement')
    plt.plot(historique['val'], label='Validation')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.title('Courbe d\'apprentissage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHEMIN_DONNEES['resultats'], 'courbe_apprentissage.png'))
    print(f"Courbe d'apprentissage sauvegardée dans {os.path.join(CHEMIN_DONNEES['resultats'], 'courbe_apprentissage.png')}")
    
    # Évaluer le modèle sur l'ensemble de test
    metriques = evaluer_modele(modele, test_loader)
    
    # Afficher les métriques
    print("\nMétriques d'évaluation sur l'ensemble de test:")
    for metrique, valeur in metriques.items():
        print(f"  {metrique}: {valeur:.4f}")


def evaluer(args):
    """
    Évalue le modèle d'optimisation d'irrigation.
    
    Args:
        args (Namespace): Arguments de ligne de commande
    """
    print("Démarrage de l'évaluation du modèle...")
    
    # Créer les dataloaders
    _, _, test_loader = creer_dataloaders(
        repertoire_images=args.repertoire_images,
        taille_sequence=args.taille_sequence,
        taille_lot=args.taille_lot
    )
    
    # Créer le modèle
    modele = ModeleIrrigationOptimisation()
    
    # Charger les poids du modèle
    chemin_modele = args.chemin_modele or os.path.join(CHEMIN_DONNEES['modeles'], 'meilleur_modele.pth')
    if os.path.exists(chemin_modele):
        modele.load_state_dict(torch.load(chemin_modele))
        print(f"Modèle chargé depuis {chemin_modele}")
    else:
        print(f"Erreur: Le fichier modèle {chemin_modele} n'existe pas.")
        return
    
    # Utiliser le GPU si disponible
    if torch.cuda.is_available():
        modele = modele.cuda()
        print("Utilisation du GPU pour l'évaluation.")
    
    # Évaluer le modèle
    metriques = evaluer_modele(modele, test_loader)
    
    # Afficher les métriques
    print("\nMétriques d'évaluation:")
    for metrique, valeur in metriques.items():
        print(f"  {metrique}: {valeur:.4f}")


def inferer(args):
    """
    Effectue l'inférence avec le modèle d'optimisation d'irrigation.
    
    Args:
        args (Namespace): Arguments de ligne de commande
    """
    print("Démarrage de l'inférence avec le modèle...")
    
    # Créer le modèle
    modele = ModeleIrrigationOptimisation()
    
    # Charger les poids du modèle
    chemin_modele = args.chemin_modele or os.path.join(CHEMIN_DONNEES['modeles'], 'meilleur_modele.pth')
    if os.path.exists(chemin_modele):
        modele.load_state_dict(torch.load(chemin_modele))
        print(f"Modèle chargé depuis {chemin_modele}")
    else:
        print(f"Erreur: Le fichier modèle {chemin_modele} n'existe pas.")
        return
    
    # Utiliser le GPU si disponible
    if torch.cuda.is_available():
        modele = modele.cuda()
        print("Utilisation du GPU pour l'inférence.")
    
    # Mode évaluation
    modele.eval()
    
    # Charger les données d'entrée (ici, nous utilisons des données fictives pour l'exemple)
    from deep_learning.data_pipeline import SequenceTemporelleDataset
    import torchvision.transforms as transforms
    
    # Définir les transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Créer un dataset pour les données d'entrée
    dataset = SequenceTemporelleDataset(
        repertoire_base=args.repertoire_images,
        liste_dates=[f"A{args.date.replace('-', '')}"],  # Format: AAAAMMJJ
        taille_sequence=args.taille_sequence,
        transform=transform,
        mode='test'
    )
    
    # Vérifier si le dataset contient des données
    if len(dataset) == 0:
        print("Erreur: Aucune donnée disponible pour la date spécifiée.")
        return
    
    # Obtenir les données d'entrée
    sequences, metadata = dataset[0]
    
    # Ajouter une dimension batch
    sequences = {var: seq.unsqueeze(0) for var, seq in sequences.items()}
    
    # Mettre les données sur le GPU si disponible
    if torch.cuda.is_available():
        sequences = {var: seq.cuda() for var, seq in sequences.items()}
    
    # Effectuer l'inférence
    with torch.no_grad():
        predictions = modele(sequences)
    
    # Formater les prédictions
    predictions_formattees = modele._formater_predictions(predictions)
    
    # Générer les recommandations
    date_obj = datetime.strptime(args.date, "%Y-%m-%d")
    recommandations = modele.generer_recommandations(
        predictions=predictions_formattees,
        latitude=args.latitude,
        longitude=args.longitude,
        date=date_obj,
        utilisation_actuelle=args.utilisation_actuelle,
        contraintes_eau=args.contraintes_eau
    )
    
    # Afficher les recommandations
    import json
    print("\nRecommandations:")
    print(json.dumps(recommandations, indent=2, ensure_ascii=False))
    
    # Sauvegarder les recommandations
    chemin_sortie = os.path.join(CHEMIN_DONNEES['resultats'], f"recommandations_{args.date}.json")
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        json.dump(recommandations, f, indent=2, ensure_ascii=False)
    
    print(f"Recommandations sauvegardées dans {chemin_sortie}")


def main():
    """
    Fonction principale qui analyse les arguments et exécute la commande appropriée.
    """
    # Créer le parseur d'arguments
    parser = argparse.ArgumentParser(description="Modèle de Deep Learning pour l'optimisation d'irrigation")
    subparsers = parser.add_subparsers(dest='commande', help='Commande à exécuter')
    
    # Parseur pour la commande 'entrainer'
    parser_entrainer = subparsers.add_parser('entrainer', help="Entraîner le modèle")
    parser_entrainer.add_argument('--repertoire-images', type=str, default=CHEMIN_DONNEES['images'],
                                help="Chemin vers le répertoire des images")
    parser_entrainer.add_argument('--taille-sequence', type=int, default=30,
                                help="Nombre d'images dans une séquence")
    parser_entrainer.add_argument('--taille-lot', type=int, default=32,
                                help="Taille du batch pour l'entraînement")
    parser_entrainer.add_argument('--epoques', type=int, default=100,
                                help="Nombre d'époques d'entraînement")
    parser_entrainer.add_argument('--patience', type=int, default=10,
                                help="Patience pour l'early stopping")
    parser_entrainer.add_argument('--taux-apprentissage', type=float, default=1e-4,
                                help="Taux d'apprentissage")
    
    # Parseur pour la commande 'evaluer'
    parser_evaluer = subparsers.add_parser('evaluer', help="Évaluer le modèle")
    parser_evaluer.add_argument('--repertoire-images', type=str, default=CHEMIN_DONNEES['images'],
                              help="Chemin vers le répertoire des images")
    parser_evaluer.add_argument('--taille-sequence', type=int, default=30,
                              help="Nombre d'images dans une séquence")
    parser_evaluer.add_argument('--taille-lot', type=int, default=32,
                              help="Taille du batch pour l'évaluation")
    parser_evaluer.add_argument('--chemin-modele', type=str, default=None,
                              help="Chemin vers le fichier modèle à charger")
    
    # Parseur pour la commande 'inferer'
    parser_inferer = subparsers.add_parser('inferer', help="Effectuer l'inférence avec le modèle")
    parser_inferer.add_argument('--repertoire-images', type=str, default=CHEMIN_DONNEES['images'],
                              help="Chemin vers le répertoire des images")
    parser_inferer.add_argument('--taille-sequence', type=int, default=30,
                              help="Nombre d'images dans une séquence")
    parser_inferer.add_argument('--chemin-modele', type=str, default=None,
                              help="Chemin vers le fichier modèle à charger")
    parser_inferer.add_argument('--date', type=str, required=True,
                              help="Date pour laquelle faire l'inférence (format: YYYY-MM-DD)")
    parser_inferer.add_argument('--latitude', type=float, required=True,
                              help="Latitude des coordonnées géographiques")
    parser_inferer.add_argument('--longitude', type=float, required=True,
                              help="Longitude des coordonnées géographiques")
    parser_inferer.add_argument('--utilisation-actuelle', type=str, default=None,
                              help="Informations sur l'utilisation actuelle des terres (format JSON)")
    parser_inferer.add_argument('--contraintes-eau', type=str, default=None,
                              help="Contraintes d'eau disponible (format JSON)")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Créer les dossiers nécessaires
    creer_dossiers_necessaires()
    
    # Exécuter la commande appropriée
    if args.commande == 'entrainer':
        entrainer(args)
    elif args.commande == 'evaluer':
        evaluer(args)
    elif args.commande == 'inferer':
        # Convertir les arguments JSON en dictionnaires
        import json
        if args.utilisation_actuelle:
            args.utilisation_actuelle = json.loads(args.utilisation_actuelle)
        if args.contraintes_eau:
            args.contraintes_eau = json.loads(args.contraintes_eau)
        
        inferer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()