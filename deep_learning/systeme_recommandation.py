# -*- coding: utf-8 -*-
"""
Système de recommandation pour l'optimisation d'irrigation

Ce module implémente le système de recommandation qui utilise les prédictions
du modèle de Deep Learning pour fournir des recommandations d'irrigation,
d'analyse du sol et de cultures adaptées.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.config import CONFIG_RECOMMANDATION


class AnalyseSol:
    """
    Classe pour l'analyse du sol basée sur les prédictions du modèle.
    """
    def __init__(self):
        """
        Initialise l'analyseur de sol.
        """
        # Dictionnaire des types de sol et leurs caractéristiques
        self.types_sol = {
            'sableux': {
                'capacite_retention': 'faible',
                'drainage': 'rapide',
                'nutriments': 'faible'
            },
            'limoneux': {
                'capacite_retention': 'moyenne',
                'drainage': 'modéré',
                'nutriments': 'moyenne'
            },
            'argileux': {
                'capacite_retention': 'élevée',
                'drainage': 'lent',
                'nutriments': 'élevée'
            },
            'loam': {
                'capacite_retention': 'bonne',
                'drainage': 'bon',
                'nutriments': 'bonne'
            },
            'organique': {
                'capacite_retention': 'très élevée',
                'drainage': 'variable',
                'nutriments': 'très élevée'
            }
        }
    
    def analyser(self, humidite_sol, texture_sol_pred, nutriments_pred):
        """
        Analyse le sol en fonction des prédictions du modèle.
        
        Args:
            humidite_sol (float): Valeur d'humidité du sol prédite
            texture_sol_pred (str): Type de sol prédit
            nutriments_pred (dict): Prédictions des niveaux de nutriments
            
        Returns:
            dict: Résultats de l'analyse du sol
        """
        # Déterminer le type de sol
        type_sol = texture_sol_pred
        
        # Obtenir les caractéristiques du type de sol
        caracteristiques = self.types_sol.get(type_sol, self.types_sol['limoneux'])
        
        # Calculer la capacité de rétention d'eau en mm/m
        capacites_retention = {
            'sableux': 70,
            'limoneux': 150,
            'argileux': 200,
            'loam': 180,
            'organique': 250
        }
        capacite_retention_mm = capacites_retention.get(type_sol, 150)
        
        # Estimer la teneur en nutriments
        nutriments = {
            'azote': nutriments_pred.get('azote', 'moyen'),
            'phosphore': nutriments_pred.get('phosphore', 'moyen'),
            'potassium': nutriments_pred.get('potassium', 'moyen'),
            'matiere_organique': nutriments_pred.get('matiere_organique', 'moyen')
        }
        
        # Créer le rapport d'analyse
        analyse = {
            'type_sol': type_sol,
            'caracteristiques': caracteristiques,
            'capacite_retention_mm': capacite_retention_mm,
            'humidite_actuelle': humidite_sol,
            'nutriments': nutriments,
            'recommandations_amelioration': self._generer_recommandations_amelioration(type_sol, nutriments)
        }
        
        return analyse
    
    def _generer_recommandations_amelioration(self, type_sol, nutriments):
        """
        Génère des recommandations pour améliorer la qualité du sol.
        
        Args:
            type_sol (str): Type de sol
            nutriments (dict): Niveaux de nutriments
            
        Returns:
            list: Liste de recommandations
        """
        recommandations = []
        
        # Recommandations basées sur le type de sol
        if type_sol == 'sableux':
            recommandations.append("Ajouter de la matière organique pour améliorer la rétention d'eau.")
            recommandations.append("Utiliser du compost ou du fumier pour augmenter la teneur en nutriments.")
        elif type_sol == 'argileux':
            recommandations.append("Ajouter du sable ou de la matière organique pour améliorer le drainage.")
            recommandations.append("Éviter de travailler le sol lorsqu'il est trop humide.")
        elif type_sol == 'limoneux':
            recommandations.append("Maintenir un bon niveau de matière organique pour préserver la structure du sol.")
        
        # Recommandations basées sur les nutriments
        if nutriments['azote'] == 'faible':
            recommandations.append("Ajouter un engrais riche en azote ou planter des légumineuses.")
        if nutriments['phosphore'] == 'faible':
            recommandations.append("Incorporer du phosphate de roche ou du fumier composté.")
        if nutriments['potassium'] == 'faible':
            recommandations.append("Ajouter de la cendre de bois ou un engrais potassique.")
        if nutriments['matiere_organique'] == 'faible':
            recommandations.append("Incorporer du compost, du fumier ou utiliser des cultures de couverture.")
        
        return recommandations


class RecommandationCultures:
    """
    Classe pour la recommandation de cultures adaptées aux conditions locales.
    """
    def __init__(self):
        """
        Initialise le recommandeur de cultures.
        """
        # Dictionnaire des cultures et leurs exigences
        self.cultures = {
            'maïs': {
                'besoin_eau': 'élevé',
                'tolerance_secheresse': 'moyenne',
                'type_sol_prefere': ['limoneux', 'loam'],
                'temperature_optimale': (20, 30),  # °C
                'saison_plantation': ['printemps'],
                'duree_croissance': 90  # jours
            },
            'blé': {
                'besoin_eau': 'moyen',
                'tolerance_secheresse': 'bonne',
                'type_sol_prefere': ['limoneux', 'argileux', 'loam'],
                'temperature_optimale': (15, 25),
                'saison_plantation': ['automne', 'hiver'],
                'duree_croissance': 120
            },
            'riz': {
                'besoin_eau': 'très élevé',
                'tolerance_secheresse': 'très faible',
                'type_sol_prefere': ['argileux'],
                'temperature_optimale': (20, 35),
                'saison_plantation': ['printemps', 'été'],
                'duree_croissance': 120
            },
            'soja': {
                'besoin_eau': 'moyen',
                'tolerance_secheresse': 'moyenne',
                'type_sol_prefere': ['limoneux', 'loam'],
                'temperature_optimale': (20, 30),
                'saison_plantation': ['printemps'],
                'duree_croissance': 100
            },
            'tomate': {
                'besoin_eau': 'moyen',
                'tolerance_secheresse': 'moyenne',
                'type_sol_prefere': ['loam', 'sableux'],
                'temperature_optimale': (20, 30),
                'saison_plantation': ['printemps', 'été'],
                'duree_croissance': 70
            },
            'pomme de terre': {
                'besoin_eau': 'moyen',
                'tolerance_secheresse': 'faible',
                'type_sol_prefere': ['loam', 'sableux'],
                'temperature_optimale': (15, 25),
                'saison_plantation': ['printemps'],
                'duree_croissance': 90
            },
            'coton': {
                'besoin_eau': 'élevé',
                'tolerance_secheresse': 'bonne',
                'type_sol_prefere': ['loam', 'argileux'],
                'temperature_optimale': (25, 35),
                'saison_plantation': ['printemps'],
                'duree_croissance': 150
            },
            'canne à sucre': {
                'besoin_eau': 'élevé',
                'tolerance_secheresse': 'moyenne',
                'type_sol_prefere': ['loam', 'argileux'],
                'temperature_optimale': (25, 35),
                'saison_plantation': ['printemps', 'été'],
                'duree_croissance': 300
            }
        }
    
    def recommander(self, analyse_sol, temperature, precipitation, date, contraintes_eau=None):
        """
        Recommande des cultures adaptées aux conditions locales.
        
        Args:
            analyse_sol (dict): Résultats de l'analyse du sol
            temperature (float): Température moyenne prévue (°C)
            precipitation (float): Précipitations moyennes prévues (mm)
            date (datetime): Date pour laquelle faire la recommandation
            contraintes_eau (dict, optional): Contraintes d'eau disponible
            
        Returns:
            list: Liste des cultures recommandées avec leurs scores
        """
        # Déterminer la saison en fonction de la date
        saison = self._determiner_saison(date)
        
        # Calculer un score pour chaque culture
        scores_cultures = {}
        for culture, caracteristiques in self.cultures.items():
            score = self._calculer_score_culture(
                culture, caracteristiques, analyse_sol, temperature, precipitation, saison, contraintes_eau)
            scores_cultures[culture] = score
        
        # Trier les cultures par score décroissant
        cultures_triees = sorted(scores_cultures.items(), key=lambda x: x[1], reverse=True)
        
        # Préparer les recommandations détaillées pour les 3 meilleures cultures
        recommandations = []
        for culture, score in cultures_triees[:3]:
            caract = self.cultures[culture]
            recommandation = {
                'culture': culture,
                'score_adequation': score,
                'rendement_estime': self._estimer_rendement(culture, score),
                'date_plantation': self._calculer_date_plantation(culture, date),
                'date_recolte': self._calculer_date_recolte(culture, date),
                'besoins_eau': caract['besoin_eau'],
                'temperature_optimale': caract['temperature_optimale']
            }
            recommandations.append(recommandation)
        
        return recommandations
    
    def _determiner_saison(self, date):
        """
        Détermine la saison en fonction de la date.
        
        Args:
            date (datetime): Date
            
        Returns:
            str: Saison ('printemps', 'été', 'automne', 'hiver')
        """
        mois = date.month
        if 3 <= mois <= 5:
            return 'printemps'
        elif 6 <= mois <= 8:
            return 'été'
        elif 9 <= mois <= 11:
            return 'automne'
        else:
            return 'hiver'
    
    def _calculer_score_culture(self, culture, caracteristiques, analyse_sol, temperature, precipitation, saison, contraintes_eau):
        """
        Calcule un score d'adéquation pour une culture.
        
        Args:
            culture (str): Nom de la culture
            caracteristiques (dict): Caractéristiques de la culture
            analyse_sol (dict): Résultats de l'analyse du sol
            temperature (float): Température moyenne prévue (°C)
            precipitation (float): Précipitations moyennes prévues (mm)
            saison (str): Saison actuelle
            contraintes_eau (dict): Contraintes d'eau disponible
            
        Returns:
            float: Score d'adéquation (0-100)
        """
        score = 0
        
        # Score pour le type de sol
        if analyse_sol['type_sol'] in caracteristiques['type_sol_prefere']:
            score += 20
        else:
            score += 5
        
        # Score pour la température
        temp_min, temp_max = caracteristiques['temperature_optimale']
        if temp_min <= temperature <= temp_max:
            score += 20
        else:
            # Pénalité proportionnelle à l'écart de température
            ecart = min(abs(temperature - temp_min), abs(temperature - temp_max))
            score += max(0, 20 - ecart * 2)
        
        # Score pour la saison de plantation
        if saison in caracteristiques['saison_plantation']:
            score += 20
        else:
            score += 0
        
        # Score pour les besoins en eau vs précipitations et contraintes
        besoins_eau = {
            'très faible': 1,
            'faible': 2,
            'moyen': 3,
            'élevé': 4,
            'très élevé': 5
        }
        
        besoin = besoins_eau.get(caracteristiques['besoin_eau'], 3)
        
        # Évaluer si les précipitations sont suffisantes
        precipitations_suffisantes = False
        if besoin == 1 and precipitation > 10:
            precipitations_suffisantes = True
        elif besoin == 2 and precipitation > 20:
            precipitations_suffisantes = True
        elif besoin == 3 and precipitation > 30:
            precipitations_suffisantes = True
        elif besoin == 4 and precipitation > 40:
            precipitations_suffisantes = True
        elif besoin == 5 and precipitation > 50:
            precipitations_suffisantes = True
        
        if precipitations_suffisantes:
            score += 20
        else:
            # Si les précipitations ne sont pas suffisantes, vérifier les contraintes d'eau
            if contraintes_eau and contraintes_eau.get('disponibilite', 'normale') == 'limitée':
                # Favoriser les cultures à faible besoin en eau
                if besoin <= 2:
                    score += 15
                elif besoin == 3:
                    score += 10
                else:
                    score += 5
            else:
                # Sans contraintes, score proportionnel à l'adéquation entre besoin et précipitations
                score += max(0, 20 - (besoin * 10 - precipitation) * 0.2)
        
        # Score pour la tolérance à la sécheresse si les précipitations sont faibles
        if precipitation < 30:
            tolerance = {
                'très faible': 0,
                'faible': 5,
                'moyenne': 10,
                'bonne': 15,
                'excellente': 20
            }
            score += tolerance.get(caracteristiques['tolerance_secheresse'], 10)
        
        # Normaliser le score sur 100
        score = min(100, score)
        
        return score
    
    def _estimer_rendement(self, culture, score):
        """
        Estime le rendement potentiel d'une culture en fonction de son score d'adéquation.
        
        Args:
            culture (str): Nom de la culture
            score (float): Score d'adéquation
            
        Returns:
            dict: Estimation du rendement
        """
        # Rendements de référence en tonnes/hectare
        rendements_reference = {
            'maïs': 10.0,
            'blé': 8.0,
            'riz': 7.0,
            'soja': 3.5,
            'tomate': 80.0,
            'pomme de terre': 40.0,
            'coton': 2.5,
            'canne à sucre': 100.0
        }
        
        rendement_ref = rendements_reference.get(culture, 5.0)
        
        # Calculer le rendement estimé en fonction du score
        rendement_estime = rendement_ref * (score / 100)
        
        # Ajouter une marge d'incertitude
        marge = rendement_estime * 0.2
        
        return {
            'min': round(rendement_estime - marge, 1),
            'max': round(rendement_estime + marge, 1),
            'unite': 'tonnes/hectare'
        }
    
    def _calculer_date_plantation(self, culture, date_reference):
        """
        Calcule la date de plantation recommandée.
        
        Args:
            culture (str): Nom de la culture
            date_reference (datetime): Date de référence
            
        Returns:
            str: Date de plantation recommandée
        """
        # Logique simplifiée pour déterminer la date de plantation
        # Dans un système réel, cela serait beaucoup plus complexe
        saison = self._determiner_saison(date_reference)
        caract = self.cultures[culture]
        
        if saison in caract['saison_plantation']:
            # Si la saison actuelle est bonne pour planter
            return date_reference.strftime("%d/%m/%Y")
        else:
            # Sinon, recommander la prochaine saison de plantation
            saisons = ['hiver', 'printemps', 'été', 'automne']
            saison_actuelle_idx = saisons.index(saison)
            
            for i in range(1, 5):  # Vérifier les 4 prochaines saisons
                prochaine_saison = saisons[(saison_actuelle_idx + i) % 4]
                if prochaine_saison in caract['saison_plantation']:
                    # Calculer une date approximative pour cette saison
                    mois_debut = {
                        'printemps': 3,
                        'été': 6,
                        'automne': 9,
                        'hiver': 12
                    }
                    nouvelle_date = date_reference.replace(month=mois_debut[prochaine_saison], day=1)
                    if nouvelle_date < date_reference:
                        nouvelle_date = nouvelle_date.replace(year=date_reference.year + 1)
                    return nouvelle_date.strftime("%d/%m/%Y")
            
            # Si aucune saison ne convient (ne devrait pas arriver)
            return "Date non déterminée"
    
    def _calculer_date_recolte(self, culture, date_plantation):
        """
        Calcule la date de récolte estimée.
        
        Args:
            culture (str): Nom de la culture
            date_plantation (datetime): Date de plantation
            
        Returns:
            str: Date de récolte estimée
        """
        # Ajouter la durée de croissance à la date de plantation
        from datetime import timedelta
        duree = self.cultures[culture]['duree_croissance']
        
        # Si date_plantation est une chaîne, la convertir en datetime
        if isinstance(date_plantation, str):
            date_plantation = datetime.strptime(date_plantation, "%d/%m/%Y")
        
        date_recolte = date_plantation + timedelta(days=duree)
        return date_recolte.strftime("%d/%m/%Y")


class StrategieIrrigation:
    """
    Classe pour la recommandation de stratégies d'irrigation optimales.
    """
    def __init__(self):
        """
        Initialise le recommandeur de stratégies d'irrigation.
        """
        # Méthodes d'irrigation disponibles
        self.methodes = CONFIG_RECOMMANDATION['methodes_irrigation']
        
        # Seuils de stress hydrique
        self.seuils_stress = CONFIG_RECOMMANDATION['seuils_stress_hydrique']
        
        # Efficacité des méthodes d'irrigation (pourcentage d'eau qui atteint effectivement les plantes)
        self.efficacite_methodes = {
            'goutte-à-goutte': 0.95,  # 95% efficace
            'aspersion': 0.75,       # 75% efficace
            'inondation': 0.50        # 50% efficace
        }
    
    def recommander(self, culture, analyse_sol, stress_hydrique_predit, evapotranspiration, precipitation):
        """
        Recommande une stratégie d'irrigation optimale.
        
        Args:
            culture (str): Culture pour laquelle faire la recommandation
            analyse_sol (dict): Résultats de l'analyse du sol
            stress_hydrique_predit (float): Niveau de stress hydrique prédit (0-1)
            evapotranspiration (float): Taux d'évapotranspiration prévu (mm/jour)
            precipitation (float): Précipitations prévues (mm/jour)
            
        Returns:
            dict: Stratégie d'irrigation recommandée
        """
        # Déterminer la méthode d'irrigation optimale
        methode = self._selectionner_methode(culture, analyse_sol['type_sol'], stress_hydrique_predit)
        
        # Calculer les besoins en eau
        besoin_eau = self._calculer_besoin_eau(culture, evapotranspiration, precipitation)
        
        # Ajuster en fonction de l'efficacité de la méthode
        quantite_eau = besoin_eau / self.efficacite_methodes.get(methode, 0.7)
        
        # Déterminer la fréquence et la durée d'irrigation
        frequence, duree = self._calculer_frequence_duree(methode, quantite_eau, analyse_sol)
        
        # Estimer les économies d'eau par rapport à une méthode traditionnelle
        economie_eau = self._estimer_economie_eau(methode, quantite_eau)
        
        # Créer la recommandation
        recommandation = {
            'methode': methode,
            'quantite_eau': round(quantite_eau, 1),  # litres/m²
            'frequence': frequence,
            'duree': duree,
            'economie_eau': economie_eau,
            'conseils_specifiques': self._generer_conseils(methode, culture, analyse_sol)
        }
        
        return recommandation
    
    def _selectionner_methode(self, culture, type_sol, stress_hydrique):
        """
        Sélectionne la méthode d'irrigation la plus adaptée.
        
        Args:
            culture (str): Type de culture
            type_sol (str): Type de sol
            stress_hydrique (float): Niveau de stress hydrique prédit
            
        Returns:
            str: Méthode d'irrigation recommandée
        """
        # Cultures qui bénéficient particulièrement de certaines méthodes
        preferences_cultures = {
            'tomate': 'goutte-à-goutte',
            'pomme de terre': 'goutte-à-goutte',
            'maïs': 'aspersion',
            'riz': 'inondation',
            'canne à sucre': 'goutte-à-goutte'
        }
        
        # Si la culture a une préférence spécifique, l'utiliser
        if culture in preferences_cultures:
            return preferences_cultures[culture]
        
        # Sinon, baser la décision sur le type de sol et le stress hydrique
        if type_sol == 'sableux':
            # Sol sableux a un drainage rapide, préférer goutte-à-goutte
            return 'goutte-à-goutte'
        elif type_sol == 'argileux':
            # Sol argileux retient l'eau, éviter l'inondation
            return 'aspersion' if stress_hydrique < self.seuils_stress['modere'] else 'goutte-à-goutte'
        else:
            # Pour les autres types de sol, baser sur le stress hydrique
            if stress_hydrique < self.seuils_stress['faible']:
                return 'aspersion'
            elif stress_hydrique < self.seuils_stress['modere']:
                return 'aspersion'
            else:
                return 'goutte-à-goutte'
    
    def _calculer_besoin_eau(self, culture, evapotranspiration, precipitation):
        """
        Calcule le besoin en eau quotidien.
        
        Args:
            culture (str): Type de culture
            evapotranspiration (float): Taux d'évapotranspiration (mm/jour)
            precipitation (float): Précipitations (mm/jour)
            
        Returns:
            float: Besoin en eau (litres/m²/jour)
        """
        # Coefficients culturaux (Kc) pour différentes cultures
        kc = {
            'maïs': 1.2,
            'blé': 1.15,
            'riz': 1.3,
            'soja': 1.15,
            'tomate': 1.25,
            'pomme de terre': 1.15,
            'coton': 1.2,
            'canne à sucre': 1.25
        }
        
        # Utiliser un coefficient par défaut si la culture n'est pas dans la liste
        coefficient = kc.get(culture, 1.0)
        
        # Calculer le besoin en eau (ETc = ETo * Kc)
        besoin_eau = evapotranspiration * coefficient
        
        # Soustraire les précipitations efficaces (80% des précipitations)
        precipitation_efficace = precipitation * 0.8
        besoin_net = max(0, besoin_eau - precipitation_efficace)
        
        # Convertir de mm/jour à litres/m²/jour (1 mm = 1 litre/m²)
        return besoin_net
    
    def _calculer_frequence_duree(self, methode, quantite_eau, analyse_sol):
        """
        Calcule la fréquence et la durée d'irrigation recommandées.
        
        Args:
            methode (str): Méthode d'irrigation
            quantite_eau (float): Quantité d'eau nécessaire (litres/m²/jour)
            analyse_sol (dict): Résultats de l'analyse du sol
            
        Returns:
            tuple: (fréquence, durée)
        """
        # Capacité de rétention d'eau du sol
        capacite_retention = analyse_sol['capacite_retention_mm']
        
        # Fréquence d'irrigation en fonction de la méthode et du type de sol
        if methode == 'goutte-à-goutte':
            # Irrigation fréquente mais en petites quantités
            if analyse_sol['type_sol'] == 'sableux':
                frequence = "quotidienne"
                duree = f"{int(quantite_eau * 4 * 60)} minutes"
            else:
                frequence = "tous les 7 jours"
                duree = f"{int(quantite_eau * 7 * 60)} minutes"
        
        return frequence, duree
    
    def _estimer_economie_eau(self, methode, quantite_eau):
        """
        Estime les économies d'eau par rapport à une méthode traditionnelle.
        
        Args:
            methode (str): Méthode d'irrigation
            quantite_eau (float): Quantité d'eau nécessaire (litres/m²/jour)
            
        Returns:
            dict: Économies d'eau estimées
        """
        # Méthode traditionnelle (inondation non optimisée)
        efficacite_traditionnelle = 0.4  # 40% efficace
        quantite_traditionnelle = quantite_eau / self.efficacite_methodes.get(methode, 0.7) * efficacite_traditionnelle
        
        # Calculer l'économie
        economie_absolue = quantite_traditionnelle - quantite_eau
        economie_pourcentage = (economie_absolue / quantite_traditionnelle) * 100
        
        return {
            'pourcentage': round(economie_pourcentage, 1),
            'litres_par_m2_par_jour': round(economie_absolue, 1)
        }
    
    def _generer_conseils(self, methode, culture, analyse_sol):
        """
        Génère des conseils spécifiques pour la méthode d'irrigation.
        
        Args:
            methode (str): Méthode d'irrigation
            culture (str): Type de culture
            analyse_sol (dict): Résultats de l'analyse du sol
            
        Returns:
            list: Liste de conseils
        """
        conseils = []
        
        # Conseils généraux
        conseils.append("Irriguer tôt le matin ou tard le soir pour réduire l'évaporation.")
        
        # Conseils spécifiques à la méthode
        if methode == 'goutte-à-goutte':
            conseils.append("Vérifier régulièrement les goutteurs pour éviter les obstructions.")
            conseils.append("Maintenir une pression constante dans le système pour une distribution uniforme.")
            if analyse_sol['type_sol'] == 'argileux':
                conseils.append("Réduire le débit pour éviter le ruissellement sur sol argileux.")
        
        elif methode == 'aspersion':
            conseils.append("Éviter d'irriguer par temps venteux pour minimiser les pertes par dérive.")
            conseils.append("Vérifier l'uniformité de la distribution d'eau sur toute la zone.")
            if culture in ['tomate', 'pomme de terre']:
                conseils.append("Éviter de mouiller le feuillage pour réduire les risques de maladies fongiques.")
        
        else:  # inondation
            conseils.append("Niveler soigneusement le terrain pour assurer une distribution uniforme de l'eau.")
            conseils.append("Créer des sillons ou des bassins adaptés à la culture et au type de sol.")
            if analyse_sol['type_sol'] == 'sableux':
                conseils.append("Irriguer plus fréquemment mais avec des quantités moindres sur sol sableux.")
        
        # Conseils basés sur le type de sol
        if analyse_sol['type_sol'] == 'sableux':
            conseils.append("Sur sol sableux, appliquer du paillis pour réduire l'évaporation.")
        elif analyse_sol['type_sol'] == 'argileux':
            conseils.append("Sur sol argileux, irriguer lentement pour permettre une bonne infiltration.")
        
        return conseils


class SystemeRecommandation:
    """
    Système de recommandation complet intégrant l'analyse du sol, les recommandations
    de cultures et les stratégies d'irrigation.
    """
    def __init__(self):
        """
        Initialise le système de recommandation.
        """
        self.analyse_sol = AnalyseSol()
        self.recommandation_cultures = RecommandationCultures()
        self.strategie_irrigation = StrategieIrrigation()
    
    def generer_recommandations(self, latitude, longitude, date, predictions, utilisation_actuelle=None, contraintes_eau=None):
        """
        Génère des recommandations complètes basées sur les prédictions du modèle.
        
        Args:
            latitude (float): Latitude des coordonnées géographiques
            longitude (float): Longitude des coordonnées géographiques
            date (datetime): Date pour laquelle faire les recommandations
            predictions (dict): Prédictions du modèle
            utilisation_actuelle (dict, optional): Informations sur l'utilisation actuelle des terres
            contraintes_eau (dict, optional): Contraintes d'eau disponible
            
        Returns:
            dict: Recommandations complètes
        """
        # Extraire les prédictions nécessaires
        humidite_sol = predictions.get('humidite_sol', 0.5)  # Valeur par défaut si non disponible
        texture_sol = predictions.get('type_sol', 'limoneux')
        nutriments = predictions.get('nutriments', {'azote': 'moyen', 'phosphore': 'moyen', 'potassium': 'moyen', 'matiere_organique': 'moyen'})
        temperature = predictions.get('temperature', 25.0)
        precipitation = predictions.get('precipitation', 20.0)
        evapotranspiration = predictions.get('evapotranspiration', 5.0)
        stress_hydrique = predictions.get('stress_hydrique', 0.5)
        
        # Générer l'analyse du sol
        analyse = self.analyse_sol.analyser(humidite_sol, texture_sol, nutriments)
        
        # Générer les recommandations de cultures
        cultures_recommandees = self.recommandation_cultures.recommander(
            analyse, temperature, precipitation, date, contraintes_eau)
        
        # Générer les stratégies d'irrigation pour chaque culture recommandée
        strategies_irrigation = {}
        for recommandation in cultures_recommandees:
            culture = recommandation['culture']
            strategies_irrigation[culture] = self.strategie_irrigation.recommander(
                culture, analyse, stress_hydrique, evapotranspiration, precipitation)
        
        # Créer les recommandations complètes
        recommandations = {
            'coordonnees': {'latitude': latitude, 'longitude': longitude},
            'date': date.strftime("%d/%m/%Y"),
            'analyse_sol': analyse,
            'cultures_recommandees': cultures_recommandees,
            'strategies_irrigation': strategies_irrigation,
            'meteo': {
                'temperature': temperature,
                'precipitation': precipitation,
                'evapotranspiration': evapotranspiration
            }
        }
        
        # Ajouter des informations sur l'utilisation actuelle si disponibles
        if utilisation_actuelle:
            recommandations['utilisation_actuelle'] = utilisation_actuelle
            
            # Si une culture est déjà en place, ajouter des recommandations spécifiques
            if 'culture' in utilisation_actuelle:
                culture_actuelle = utilisation_actuelle['culture']
                if culture_actuelle in self.recommandation_cultures.cultures:
                    recommandations['strategie_irrigation_actuelle'] = self.strategie_irrigation.recommander(
                        culture_actuelle, analyse, stress_hydrique, evapotranspiration, precipitation)
        
        return recommandations

if __name__ == "__main__":
    # Test du système de recommandation
    print("Test du système de recommandation...")
    
    # Créer des prédictions de test
    predictions_test = {
        'humidite_sol': 0.35,
        'type_sol': 'limoneux',
        'nutriments': {
            'azote': 'moyen',
            'phosphore': 'faible',
            'potassium': 'élevé',
            'matiere_organique': 'faible'
        },
        'temperature': 22.5,
        'precipitation': 15.0,
        'evapotranspiration': 4.5,
        'stress_hydrique': 0.4
    }
    
    # Créer une date de test
    date_test = datetime(2023, 5, 15)
    
    # Créer le système de recommandation
    systeme = SystemeRecommandation()
    
    # Générer des recommandations
    recommandations = systeme.generer_recommandations(
        latitude=34.05, 
        longitude=-118.25, 
        date=date_test, 
        predictions=predictions_test,
        utilisation_actuelle={'culture': 'maïs', 'superficie': 10.0},
        contraintes_eau={'disponibilite': 'limitée', 'source': 'puits'}
    )
    
    # Afficher les recommandations
    print("Recommandations pour le", date_test.strftime("%d/%m/%Y")) 