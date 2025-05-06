# Projet Machine Learning - Prédiction du score de Macron aux élections 2022
# UMONS, 2024-2025

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as XGBRegressor
import joblib
import warnings
import os
import glob
import re
from scipy import stats
import matplotlib.ticker as mtick

# Pour ignorer les warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("Set2")

# Pour une meilleure lisibilité dans le notebook
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

print("Chargement des bibliothèques terminé")

# 1. CHARGEMENT DES DONNÉES

# Définition du chemin des données
data_path = "src/datasets/"

# Chargement des données d'entraînement et de test
train_data = pd.read_csv(os.path.join(data_path, "results_train.csv"), sep=',', encoding='utf-8')
test_data = pd.read_csv(os.path.join(data_path, "results_test.csv"), sep=',', encoding='utf-8')

# Chargement des données additionnelles
# Niveau de vie
niveau_vie = pd.read_excel(os.path.join(data_path, "Niveau_de_vie_2013_a_la_commune.xlsx"))

# Communes de France
communes_france = pd.read_csv(os.path.join(data_path, "communes-france-2022.csv"), sep=',', encoding='utf-8')

# Données d'âge
age_insee = pd.read_excel(os.path.join(data_path, "age-insee-2020.xlsx"))

# Données diverses INSEE
insee_divers = pd.read_excel(os.path.join(data_path, "MDB-INSEE-V2.xls"))

print("Chargement des données terminé")

# 2. EXPLORATION DES DONNÉES (EDA)

# 2.1 Préparation des données pour l'analyse
# Examinons d'abord les données d'entraînement
print("\nExploration des données d'entraînement:")
print(f"Shape: {train_data.shape}")
print(train_data.head())
print(train_data.dtypes)

# Variable cible: "% Voix/Ins" pour Macron
# Filtrons pour ne garder que les lignes de Macron
macron_data = train_data[train_data['Nom'] == 'MACRON'].copy()
print(f"\nDonnées Macron uniquement: {macron_data.shape}")

# Distribution de la variable cible
plt.figure(figsize=(12, 6))
sns.histplot(macron_data['% Voix/Ins'], kde=True)
plt.title('Distribution du score de Macron (% des inscrits)')
plt.axvline(macron_data['% Voix/Ins'].mean(), color='r', linestyle='--', label=f'Moyenne: {macron_data["% Voix/Ins"].mean():.2f}%')
plt.xlabel('% Voix/Ins')
plt.ylabel('Fréquence')
plt.legend()
plt.savefig('distribution_score_macron.png')
plt.show()

print(f"Statistiques descriptives de la variable cible (% Voix/Ins):")
print(macron_data['% Voix/Ins'].describe())

# Explorons la relation entre le pourcentage de voix et d'autres variables
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
sns.scatterplot(x='% Abs/Ins', y='% Voix/Ins', data=macron_data)
plt.title('Score de Macron vs Taux d\'abstention')

plt.subplot(2, 2, 2)
sns.scatterplot(x='% Blancs/Ins', y='% Voix/Ins', data=macron_data)
plt.title('Score de Macron vs Taux de votes blancs')

plt.subplot(2, 2, 3)
sns.scatterplot(x='% Nuls/Ins', y='% Voix/Ins', data=macron_data)
plt.title('Score de Macron vs Taux de votes nuls')

plt.subplot(2, 2, 4)
sns.boxplot(x='Libellé du département', y='% Voix/Ins', data=macron_data.iloc[:200])
plt.title('Distribution du score par département (échantillon)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('relations_variables.png')
plt.show()

# 2.2 Explorons les autres datasets et leur relation avec notre variable cible

# 2.2.1 Nettoyage et préparation des données complémentaires
# Renommons et uniformisons les identifiants des communes pour faciliter la fusion
niveau_vie = niveau_vie.rename(columns={
    'Code Commune': 'CodeINSEE',
    'Nom Commune': 'Commune',
    'Niveau de vie Commune': 'NiveauVieCommune',
    'Niveau de vie Département': 'NiveauVieDepartement'
})

# Assurons-nous que CodeINSEE est au format string
niveau_vie['CodeINSEE'] = niveau_vie['CodeINSEE'].astype(str).str.zfill(5)
macron_data['CodeINSEE'] = macron_data['CodeINSEE'].astype(str).str.zfill(5)
test_data['CodeINSEE'] = test_data['CodeINSEE'].astype(str).str.zfill(5)

# Préparation des données communes France
communes_france['code_insee'] = communes_france['code_insee'].astype(str).str.zfill(5)

# Préparation des données d'âge
age_insee['INSEE'] = age_insee['INSEE'].astype(str).str.zfill(5)

# Préparation des données INSEE diverses
insee_divers['CODGEO'] = insee_divers['CODGEO'].astype(str).str.zfill(5)

# 2.2.2 Fusion des datasets pour l'analyse exploratoire
# Fusionnons les données pour voir les corrélations avec le score de Macron
merged_data = macron_data[['CodeINSEE', '% Voix/Ins', 'Libellé du département']].copy()

# Fusion avec niveau de vie
merged_data = pd.merge(merged_data, niveau_vie, on='CodeINSEE', how='left')

# Fusion avec communes_france (sélection de variables pertinentes)
communes_select = communes_france[['code_insee', 'population', 'superficie_km2', 'densite', 'altitude_moyenne',
                                 'latitude_centre', 'longitude_centre', 'grille_densite']]
merged_data = pd.merge(merged_data, communes_select, left_on='CodeINSEE', right_on='code_insee', how='left')

# Fusion avec age_insee (nous allons créer des variables agrégées)
age_groups = age_insee.copy()
# Création de variables démographiques
age_groups['PctJeunes'] = (age_groups['F0-2'] + age_groups['F3-5'] + age_groups['F6-10'] + age_groups['F11-17'] +
                          age_groups['H0-2'] + age_groups['H3-5'] + age_groups['H6-10'] + age_groups['H11-17']) / \
                         (age_groups.iloc[:, 5:].sum(axis=1)) * 100
age_groups['PctAdultes'] = (age_groups['F18-24'] + age_groups['F25-39'] + age_groups['F40-54'] +
                           age_groups['H18-24'] + age_groups['H25-39'] + age_groups['H40-54']) / \
                          (age_groups.iloc[:, 5:].sum(axis=1)) * 100
age_groups['PctSeniors'] = (age_groups['F55-64'] + age_groups['F65-79'] + age_groups['F80+'] +
                           age_groups['H55-64'] + age_groups['H65-79'] + age_groups['H80+']) / \
                          (age_groups.iloc[:, 5:].sum(axis=1)) * 100
age_groups['RatioFH'] = age_groups.iloc[:, 5:15].sum(axis=1) / age_groups.iloc[:, 15:].sum(axis=1)

age_select = age_groups[['INSEE', 'PctJeunes', 'PctAdultes', 'PctSeniors', 'RatioFH']]
merged_data = pd.merge(merged_data, age_select, left_on='CodeINSEE', right_on='INSEE', how='left')

# Fusion avec insee_divers (sélection de variables pertinentes)
insee_select = insee_divers[['CODGEO', 'Moyenne Revenus Fiscaux Départementaux', 'Moyenne Revnus fiscaux',
                           'Urbanité Ruralité', 'Taux Propriété', 'Orientation Economique', 'Dynamique Démographique INSEE']]
merged_data = pd.merge(merged_data, insee_select, left_on='CodeINSEE', right_on='CODGEO', how='left')

# 2.2.3 Analyse des corrélations
print("\nAnalyse des corrélations avec le score de Macron:")

# Sélection des variables numériques pour la matrice de corrélation
numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
corr_matrix = merged_data[numeric_cols].corr()

# Affichage des corrélations les plus fortes avec la variable cible
corr_with_target = corr_matrix['% Voix/Ins'].sort_values(ascending=False)
print(corr_with_target)

# Visualisation de la matrice de corrélation
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matrice de corrélation des variables numériques')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Visualisation des relations entre variables importantes et score de Macron
plt.figure(figsize=(16, 16))
important_features = ['NiveauVieCommune', 'densite', 'PctSeniors', 'PctAdultes', 'Moyenne Revnus fiscaux']

for i, feature in enumerate(important_features):
    plt.subplot(3, 2, i+1)
    sns.scatterplot(x=feature, y='% Voix/Ins', data=merged_data, alpha=0.5)
    plt.title(f'Score de Macron vs {feature}')
    try:
        z = np.polyfit(merged_data[feature].dropna(), merged_data.loc[merged_data[feature].notna(), '% Voix/Ins'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(merged_data[feature].dropna()), p(sorted(merged_data[feature].dropna())), "r--", linewidth=2)
    except:
        pass

plt.tight_layout()
plt.savefig('important_features.png')
plt.show()

# 2.2.4. Analyse des variables catégorielles
plt.figure(figsize=(14, 10))
cat_vars = ['grille_densite', 'Urbanité Ruralité', 'Orientation Economique', 'Dynamique Démographique INSEE']

for i, var in enumerate(cat_vars):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=var, y='% Voix/Ins', data=merged_data)
    plt.title(f'Score de Macron par {var}')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('categorical_analysis.png')
plt.show()

print("\nRésumé des données manquantes par colonne:")
print(merged_data.isnull().sum().sort_values(ascending=False))

# 3. PRÉTRAITEMENT DES DONNÉES

# 3.1 Préparation des données pour la modélisation
# Fusionnons tous les datasets pour créer nos features
def prepare_datasets(train_data, test_data, niveau_vie, communes_france, age_insee, insee_divers):
    """
    Fonction pour préparer et fusionner les datasets pour la modélisation
    """
    # Extraction des données de Macron uniquement pour l'entraînement
    train_macron = train_data[train_data['Nom'] == 'MACRON'].copy()

    # Sélection des colonnes pertinentes pour l'entraînement
    train_features = train_macron[['CodeINSEE', '% Voix/Ins', 'Inscrits', 'Abstentions', '% Abs/Ins',
                                 'Votants', '% Vot/Ins', 'Blancs', '% Blancs/Ins', '% Blancs/Vot',
                                 'Nuls', '% Nuls/Ins', '% Nuls/Vot', 'Exprimés', '% Exp/Ins', '% Exp/Vot',
                                 'Libellé du département']].copy()

    # Préparation des features pour le test
    test_features = test_data[['CodeINSEE', 'Inscrits', 'Libellé du département']].copy()

    # Assurer que CodeINSEE est au format string avec padding
    train_features['CodeINSEE'] = train_features['CodeINSEE'].astype(str).str.zfill(5)
    test_features['CodeINSEE'] = test_features['CodeINSEE'].astype(str).str.zfill(5)
    niveau_vie['CodeINSEE'] = niveau_vie['CodeINSEE'].astype(str).str.zfill(5)
    communes_france['code_insee'] = communes_france['code_insee'].astype(str).str.zfill(5)
    age_insee['INSEE'] = age_insee['INSEE'].astype(str).str.zfill(5)
    insee_divers['CODGEO'] = insee_divers['CODGEO'].astype(str).str.zfill(5)

    # Fusion avec niveau de vie
    train_features = pd.merge(train_features, niveau_vie, on='CodeINSEE', how='left')
    test_features = pd.merge(test_features, niveau_vie, on='CodeINSEE', how='left')

    # Fusion avec communes_france
    communes_select = communes_france[['code_insee', 'population', 'superficie_km2', 'densite', 'altitude_moyenne',
                                     'latitude_centre', 'longitude_centre', 'grille_densite']]
    train_features = pd.merge(train_features, communes_select, left_on='CodeINSEE', right_on='code_insee', how='left')
    test_features = pd.merge(test_features, communes_select, left_on='CodeINSEE', right_on='code_insee', how='left')

    # Préparation et fusion avec age_insee
    age_groups = age_insee.copy()
    # Création de variables démographiques
    age_groups['PctJeunes'] = (age_groups['F0-2'] + age_groups['F3-5'] + age_groups['F6-10'] + age_groups['F11-17'] +
                              age_groups['H0-2'] + age_groups['H3-5'] + age_groups['H6-10'] + age_groups['H11-17']) / \
                             (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctAdultes'] = (age_groups['F18-24'] + age_groups['F25-39'] + age_groups['F40-54'] +
                               age_groups['H18-24'] + age_groups['H25-39'] + age_groups['H40-54']) / \
                              (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctSeniors'] = (age_groups['F55-64'] + age_groups['F65-79'] + age_groups['F80+'] +
                               age_groups['H55-64'] + age_groups['H65-79'] + age_groups['H80+']) / \
                              (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['RatioFH'] = age_groups.iloc[:, 5:15].sum(axis=1) / age_groups.iloc[:, 15:].sum(axis=1)
    age_groups['PctJeunes18_24'] = (age_groups['F18-24'] + age_groups['H18-24']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctAdultes25_39'] = (age_groups['F25-39'] + age_groups['H25-39']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctAdultes40_54'] = (age_groups['F40-54'] + age_groups['H40-54']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctSeniors55_64'] = (age_groups['F55-64'] + age_groups['H55-64']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctSeniors65_79'] = (age_groups['F65-79'] + age_groups['H65-79']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100
    age_groups['PctSeniors80plus'] = (age_groups['F80+'] + age_groups['H80+']) / (age_groups.iloc[:, 5:].sum(axis=1)) * 100

    age_select = age_groups[['INSEE', 'PctJeunes', 'PctAdultes', 'PctSeniors', 'RatioFH',
                            'PctJeunes18_24', 'PctAdultes25_39', 'PctAdultes40_54',
                            'PctSeniors55_64', 'PctSeniors65_79', 'PctSeniors80plus']]
    train_features = pd.merge(train_features, age_select, left_on='CodeINSEE', right_on='INSEE', how='left')
    test_features = pd.merge(test_features, age_select, left_on='CodeINSEE', right_on='INSEE', how='left')

    # Fusion avec insee_divers
    insee_select = insee_divers[['CODGEO', 'Moyenne Revenus Fiscaux Départementaux', 'Moyenne Revnus fiscaux',
                               'Urbanité Ruralité', 'Taux Propriété', 'Orientation Economique',
                               'Dynamique Démographique INSEE', 'Nb Entreprises Secteur Services',
                               'Nb Entreprises Secteur Commerce', 'Nb Entreprises Secteur Construction',
                               'Nb Entreprises Secteur Industrie', 'Taux étudiants', 'Score Urbanité',
                               'Capacité Fiscale', 'Evolution Population']]
    train_features = pd.merge(train_features, insee_select, left_on='CodeINSEE', right_on='CODGEO', how='left')
    test_features = pd.merge(test_features, insee_select, left_on='CodeINSEE', right_on='CODGEO', how='left')

    # Extraction de la variable cible
    y_train = train_features['% Voix/Ins'].copy()

    # Création d'une feature d'urbanité basée sur la densité
    train_features['log_densite'] = np.log1p(train_features['densite'])
    test_features['log_densite'] = np.log1p(test_features['densite'])

    # Création d'une catégorie de taille de commune
    def categorize_population(pop):
        if pd.isna(pop): return "Unknown"
        elif pop < 500: return "Très petite"
        elif pop < 2000: return "Petite"
        elif pop < 10000: return "Moyenne"
        elif pop < 50000: return "Grande"
        else: return "Très grande"

    train_features['taille_commune'] = train_features['population'].apply(categorize_population)
    test_features['taille_commune'] = test_features['population'].apply(categorize_population)

    # Extraction du département à partir du code INSEE
    train_features['departement'] = train_features['CodeINSEE'].str[:2]
    test_features['departement'] = test_features['CodeINSEE'].str[:2]

    # Calcul du ratio entreprises par habitant
    for df in [train_features, test_features]:
        total_entreprises = df['Nb Entreprises Secteur Services'].fillna(0) + \
                           df['Nb Entreprises Secteur Commerce'].fillna(0) + \
                           df['Nb Entreprises Secteur Construction'].fillna(0) + \
                           df['Nb Entreprises Secteur Industrie'].fillna(0)
        df['ratio_entreprises_pop'] = total_entreprises / df['population'].replace(0, np.nan)

        # Ratio services/industries
        services = df['Nb Entreprises Secteur Services'].fillna(0)
        industries = df['Nb Entreprises Secteur Industrie'].fillna(0)
        df['ratio_services_industries'] = services / industries.replace(0, np.nan)

    # Suppression des colonnes dupliquées ou inutiles pour la modélisation
    drop_cols = ['INSEE', 'code_insee', 'CODGEO', 'Commune']
    train_features = train_features.drop([col for col in drop_cols if col in train_features.columns], axis=1)
    test_features = test_features.drop([col for col in drop_cols if col in test_features.columns], axis=1)

    return train_features, y_train, test_features

# Préparation des données
train_features, y_train, test_features = prepare_datasets(
    train_data, test_data, niveau_vie, communes_france, age_insee, insee_divers
)

print("\nDimensions des features d'entraînement:", train_features.shape)
print("Dimensions des features de test:", test_features.shape)

# Vérification des données manquantes
missing_train = train_features.isnull().sum() / len(train_features) * 100
missing_test = test_features.isnull().sum() / len(test_features) * 100

print("\nPourcentage de données manquantes (train):")
print(missing_train[missing_train > 0].sort_values(ascending=False))

print("\nPourcentage de données manquantes (test):")
print(missing_test[missing_test > 0].sort_values(ascending=False))

# 3.2 Séparation des sets d'entraînement et de validation
X_train, X_val, y_train_split, y_val = train_test_split(
    train_features, y_train, test_size=0.2, random_state=42
)

print(f"\nDimensions du set d'entraînement: {X_train.shape}")
print(f"Dimensions du set de validation: {X_val.shape}")

# 3.3 Préparation des pipelines pour le prétraitement
# Identifier les colonnes numériques et catégorielles
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Supprimer CodeINSEE des features car c'est un identifiant
if 'CodeINSEE' in numeric_features:
    numeric_features.remove('CodeINSEE')
if 'CodeINSEE' in categorical_features:
    categorical_features.remove('CodeINSEE')

print(f"\nNombre de features numériques: {len(numeric_features)}")
print(f"Nombre de features catégorielles: {len(categorical_features)}")

# Création des pipelines de prétraitement
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. MODÉLISATION

# 4.1 Modèle imposé: Lasso-Ridge (ElasticNet)
# ElasticNet combine les pénalités L1 (Lasso) et L2 (Ridge)
elastic_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(random_state=42))
])

# Définition des hyperparamètres à optimiser
elastic_params = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1, 10],
    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # 0 = Ridge, 1 = Lasso
}

# Recherche des meilleurs hyperparamètres avec validation croisée
elastic_cv = GridSearchCV(
    elastic_model,
    elastic_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

print("\nEntraînement du modèle ElasticNet (Lasso-Ridge)...")
elastic_cv.fit(X_train, y_train_split)

print(f"Meilleurs paramètres pour ElasticNet: {elastic_cv.best_params_}")
print(f"Meilleur score RMSE: {-elastic_cv.best_score_:.4f}")

# Évaluation sur le set de validation
elastic_pred = elastic_cv.predict(X_val)
elastic_rmse = np.sqrt(mean_squared_error(y_val, elastic_pred))
elastic_r2 = r2_score(y_val, elastic_pred)

print(f"ElasticNet - RMSE sur validation: {elastic_rmse:.4f}")
print(f"ElasticNet - R² sur validation: {elastic_r2:.4f}")

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_val, elastic_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('ElasticNet: Prédictions vs Valeurs réelles')
plt.savefig('elasticnet_predictions.png')
plt.show()

# Analyse des résidus
residuals = y_val - elastic_pred
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title('Distribution des résidus - ElasticNet')

plt.subplot(1, 2, 2)
plt.scatter(elastic_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus vs Prédictions - ElasticNet')
plt.tight_layout()
plt.savefig('elasticnet_residuals.png')
plt.show()

# 4.2 Modèle au choix: Gradient Boosting
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Définition des hyperparamètres à optimiser
gb_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 4, 5],
    'regressor__min_samples_split': [5, 10]
}

# Recherche des meilleurs hyperparamètres avec validation croisée
gb_