import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Chargement des Données
file_path = r".\LLM\donnees_agricoles_usa_traitees.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
    exit()

print("Dataset chargé (premières lignes) :")
print(df.head())
print("\n" + "="*50 + "\n")

# 2. Filtrage "Guidé" par le rendement et l'impact économique (Optionnel)
# Nous allons conserver les lignes où le rendement ET l'impact économique sont au-dessus de la médiane.
# Vous pouvez ajuster ces seuils (ex: .quantile(0.75) pour le top 25%)
yield_threshold = df['Crop_Yield_MT_per_HA'].quantile(0.5)
economic_threshold = df['Economic_Impact_Million_USD'].quantile(0.5)

df_filtered = df[
    (df['Crop_Yield_MT_per_HA'] >= yield_threshold) &
    (df['Economic_Impact_Million_USD'] >= economic_threshold)
].copy()

if df_filtered.empty:
    print("Attention : Le filtrage par rendement/impact économique a résulté en un DataFrame vide.")
    print("Utilisation du DataFrame complet à la place.")
    df_filtered = df.copy()
else:
    print(f"Dataset après filtrage (gardant {len(df_filtered)} lignes sur {len(df)} initialement) :")
    print(df_filtered.head())
    print("\n" + "="*50 + "\n")


# 3. Définition des Variables d'entrée (X) et Cibles (y)
features_columns = ['Average_Temperature_C', 'Total_Precipitation_mm', 'Soil_Health_Index']
target_columns = ['Crop_Type', 'Adaptation_Strategies']

X = df_filtered[features_columns]
y_categorical = df_filtered[target_columns]

# 4. Encodage des Variables Cibles Catégorielles
# Nous utilisons OrdinalEncoder car RandomForest peut gérer des entiers ordonnés pour les cibles.
# Un encodeur par colonne cible est plus propre pour MultiOutputClassifier.
# Cependant, pour simplifier, nous allons encoder les deux colonnes ensemble puis les séparer.
# Une meilleure approche serait d'utiliser ColumnTransformer ou d'encoder séparément.

# Initialiser les encodeurs
crop_type_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
adaptation_strategies_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Appliquer l'encodage
y_encoded = pd.DataFrame()
y_encoded['Crop_Type'] = crop_type_encoder.fit_transform(y_categorical[['Crop_Type']]).ravel()
y_encoded['Adaptation_Strategies'] = adaptation_strategies_encoder.fit_transform(y_categorical[['Adaptation_Strategies']]).ravel()

print("Variables cibles encodées (premières lignes) :")
print(y_encoded.head())
print("\n" + "="*50 + "\n")

# 5. Division des Données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} échantillons")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} échantillons")
print("\n" + "="*50 + "\n")

# 6. Choix et Entraînement du Modèle
# Utilisation d'un RandomForestClassifier comme estimateur de base
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Utilisation de MultiOutputClassifier pour gérer les cibles multiples
model = MultiOutputClassifier(base_estimator, n_jobs=-1)

print("Entraînement du modèle...")
model.fit(X_train, y_train)
print("Modèle entraîné.")
print("\n" + "="*50 + "\n")

# 7. Évaluation du Modèle
print("Évaluation du modèle sur l'ensemble de test...")
y_pred = model.predict(X_test)

# Pour l'évaluation, il est utile de décoder les prédictions pour les comparer aux vraies valeurs
# Cependant, pour les métriques comme accuracy_score, les valeurs encodées suffisent.

# Convertir y_pred (numpy array) en DataFrame pour faciliter la comparaison par colonne
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)

print("Précision (Accuracy) par cible :")
accuracy_crop_type = accuracy_score(y_test['Crop_Type'], y_pred_df['Crop_Type'])
accuracy_adaptation_strategies = accuracy_score(y_test['Adaptation_Strategies'], y_pred_df['Adaptation_Strategies'])

print(f"  Précision pour Crop_Type: {accuracy_crop_type:.4f}")
print(f"  Précision pour Adaptation_Strategies: {accuracy_adaptation_strategies:.4f}")
print("\n" + "="*50 + "\n")

# Rapports de classification détaillés (nécessite de décoder ou de connaître les labels encodés)
# Pour un rapport plus lisible, nous allons décoder les prédictions et les vraies valeurs de test.
y_test_decoded_crop = crop_type_encoder.inverse_transform(y_test[['Crop_Type']])
y_pred_decoded_crop = crop_type_encoder.inverse_transform(y_pred_df[['Crop_Type']])

y_test_decoded_strat = adaptation_strategies_encoder.inverse_transform(y_test[['Adaptation_Strategies']])
y_pred_decoded_strat = adaptation_strategies_encoder.inverse_transform(y_pred_df[['Adaptation_Strategies']])


print("Rapport de classification pour Crop_Type:")
# Obtenir les noms de classes uniques à partir des données de test décodées pour s'assurer que tous les labels sont présents
labels_crop_type = sorted(list(pd.Series(y_test_decoded_crop.ravel()).unique()))
print(classification_report(y_test_decoded_crop, y_pred_decoded_crop, labels=labels_crop_type, zero_division=0))
print("\n" + "="*50 + "\n")

print("Rapport de classification pour Adaptation_Strategies:")
labels_adaptation_strategies = sorted(list(pd.Series(y_test_decoded_strat.ravel()).unique()))
print(classification_report(y_test_decoded_strat, y_pred_decoded_strat, labels=labels_adaptation_strategies, zero_division=0))
print("\n" + "="*50 + "\n")

# Afficher quelques prédictions vs vraies valeurs (décodées)
results_df = pd.DataFrame({
    'Vrai_Crop_Type': y_test_decoded_crop.ravel(),
    'Pred_Crop_Type': y_pred_decoded_crop.ravel(),
    'Vrai_Adaptation_Strategy': y_test_decoded_strat.ravel(),
    'Pred_Adaptation_Strategy': y_pred_decoded_strat.ravel()
})
print("Quelques prédictions vs vraies valeurs (décodées) :")
print(results_df.head(10))

# Note pour l'utilisation avec un LLM :
# Ce modèle traditionnel peut servir de base. Pour un LLM, vous pourriez :
# 1. Utiliser ce modèle pour générer des paires (conditions d'entrée textuelles, recommandations textuelles)
#    pour affiner (fine-tune) un LLM.
# 2. Directement préparer des données textuelles à partir de votre CSV filtré pour affiner un LLM,
#    où l'entrée serait une description des conditions et la sortie la recommandation.