import pandas as pd

# Chemin vers votre fichier CSV
file_path = r".\LLM\climate_change_impact_on_agriculture_2024.csv"

# Charger le fichier CSV
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
    exit()

print("DataFrame original (premières lignes) :")
print(df.head())
print("\n" + "="*50 + "\n")

# 1. Filtrer les données pour ne garder que les États-Unis (USA)
df_usa = df[df['Country'] == 'USA'].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning

if df_usa.empty:
    print("Aucune donnée trouvée pour les USA. Vérifiez la colonne 'Country' et la valeur 'USA'.")
    exit()

print("DataFrame après filtrage pour USA (premières lignes) :")
print(df_usa.head())
print("\n" + "="*50 + "\n")

# 2. Sélectionner les colonnes pertinentes
# Attributs correspondant (ou proches) à vos entrées : Average_Temperature_C, Total_Precipitation_mm, Soil_Health_Index
# Attributs de sortie à prédire : Crop_Type, Adaptation_Strategies
# Attributs pour guider l'apprentissage : Crop_Yield_MT_per_HA, Economic_Impact_Million_USD
colonnes_a_garder = [
    'Average_Temperature_C',
    'Total_Precipitation_mm',
    'Soil_Health_Index',
    'Crop_Type',
    'Adaptation_Strategies',
    'Crop_Yield_MT_per_HA',
    'Economic_Impact_Million_USD'
]

# Vérifier si toutes les colonnes à garder existent dans df_usa
colonnes_manquantes = [col for col in colonnes_a_garder if col not in df_usa.columns]
if colonnes_manquantes:
    print(f"Erreur : Les colonnes suivantes sont manquantes dans le DataFrame filtré : {colonnes_manquantes}")
    exit()

df_selectionne = df_usa[colonnes_a_garder].copy() # Utiliser .copy()

print("DataFrame après sélection des colonnes (premières lignes) :")
print(df_selectionne.head())
print("\n" + "="*50 + "\n")

# 3. Normaliser les valeurs numériques entre 0 et 1
# Identifier les colonnes numériques à normaliser
colonnes_numeriques = df_selectionne.select_dtypes(include=['number']).columns
print(f"Colonnes numériques identifiées pour la normalisation : {list(colonnes_numeriques)}")

if not list(colonnes_numeriques):
    print("Aucune colonne numérique à normaliser.")
else:
    # Appliquer la normalisation Min-Max
    for colonne in colonnes_numeriques:
        min_val = df_selectionne[colonne].min()
        max_val = df_selectionne[colonne].max()
        if max_val - min_val != 0: # Éviter la division par zéro si toutes les valeurs sont identiques
            df_selectionne[colonne] = ((df_selectionne[colonne] - min_val) / (max_val - min_val)).round(4)
        else:
            # Si toutes les valeurs sont identiques, on peut les mettre à 0 ou 0.5 par exemple,
            # ou laisser tel quel si la normalisation n'est pas critique pour une colonne constante.
            # Ici, nous les mettons à 0 si la plage est nulle.
            df_selectionne[colonne] = 0 
    print("\nDataFrame après normalisation des colonnes numériques (premières lignes) :")
    print(df_selectionne.head())
    print("\n" + "="*50 + "\n")

    # Afficher quelques statistiques pour vérifier la normalisation
    print("Statistiques descriptives après normalisation (pour les colonnes numériques) :")
    print(df_selectionne[colonnes_numeriques].describe())

# Vous pouvez maintenant sauvegarder ce DataFrame traité si nécessaire :
df_selectionne.to_csv("donnees_agricoles_usa_traitees.csv", index=False)
print("\nDataFrame traité sauvegardé dans 'donnees_agricoles_usa_traitees.csv'")
