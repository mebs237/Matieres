import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Chargement du modèle et des données de test
best_model = joblib.load('best_model.pkl')
test_data = pd.read_csv("dataset/results_test.csv", sep=',', encoding='utf-8')

# Fonction pour préparer les données comme nous l'avons fait avant l'entrainement
def prepare_test_dataset(test_data, niveau_vie, communes_france, age_insee, insee_divers):
    """
    Fonction pour préparer les données de test identiquement à l'entrainement
    """
    # Préparation des features pour le test
    test_features = test_data[['CodeINSEE', 'Inscrits', 'Libellé du département']].copy()
    
    # Assurer que CodeINSEE est au format string avec padding
    test_features['CodeINSEE'] = test_features['CodeINSEE'].astype(str).str.zfill(5)
    niveau_vie['CodeINSEE'] = niveau_vie['CodeINSEE'].astype(str).str.zfill(5)
    communes_france['code_insee'] = communes_france['code_insee'].astype(str).str.zfill(5)
    age_insee['INSEE'] = age_insee['INSEE'].astype(str).str.zfill(5)
    insee_divers['CODGEO'] = insee_divers['CODGEO'].astype(str).str.zfill(5)
    
    # Fusion avec niveau de vie
    test_features = pd.merge(test_features, niveau_vie, on='CodeINSEE', how='left')
    
    # Fusion avec communes_france
    communes_select = communes_france[['code_insee', 'population', 'superficie_km2', 'densite', 'altitude_moyenne', 
                                    'latitude_centre', 'longitude_centre', 'grille_densite']]
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
                               age_groups['H55-64'] + age_