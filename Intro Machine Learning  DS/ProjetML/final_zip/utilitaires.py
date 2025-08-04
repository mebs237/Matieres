"""
    modules de fonctions utilitaires utiliser pour l'exploration et le prétraitement des données
"""

# @title manipulation des vecteurs
import pandas as pd
import numpy as np
from typing import List, Literal , Optional , Tuple
# @title création des graphiques
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from tqdm import tqdm

# @title prétraitement des données en masse
from sklearn.preprocessing import OneHotEncoder, RobustScaler , FunctionTransformer , LabelEncoder
from sklearn.compose import ColumnTransformer , make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# @title selection des features
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


# @title sélection du meilleur modèle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from typing import Dict, Tuple , List
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV


# @title fonction de score et evaluation
from scipy import stats
from scipy.stats import uniform, randint
from sklearn.metrics import root_mean_squared_error, r2_score ,mean_absolute_error , mean_squared_error


# @title initialisation des modèles
from sklearn.linear_model import ElasticNet , Lasso
from xgboost import XGBRegressor

# @title options sytèmes
import os
import sys
import joblib
import json
import warnings
import glob
import re
from IPython.display import Markdown, display
from os.path import join

#Pour ignorer les warnings
warnings.filterwarnings('ignore')

# configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('Set2')

# Pour une meilleure lisibilité dans le notebook
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.3f}'.format)


#=====================================================================

# @title Paramètres globaux
MISSINGTHRESHOLD = 70 # @param {"type":"integer"}
RANDOM_STATE = 42 # @param {"type":"integer"}
IDs= [] # @param {type:"raw"}


#=====================================================================

def sep(lg=90):
    """Affiche une ligne de séparation"""
    print("\n" + "-"*lg + "\n")

def sub(l1,l2):
  """Retourne la liste des éléments de l1 qui ne sont pas dans l2"""
  return list(set(l1) - set(l2))

def inter(l1,l2):
    """Retourne la liste des éléments communs entre l1 et l2"""
    return list(set(l1) & set(l2))

#=====================================================================


def to_missing_values(df:pd.DataFrame, threshold:int=MISSINGTHRESHOLD)->List:
    """
    Identifie les colonnes ayant un pourcentage de valeurs manquantes trop élevé (supérieur au seuil spécifié)

    Args:
        df (pandas.DataFrame): Le DataFrame à analyser
        threshold (float): Le seuil en pourcentage (entre 0 et 100) au-delà duquel une colonne est considérée
                          comme ayant trop de valeurs manquantes. Par défaut : 50

    Returns:
        list: Liste des noms de colonnes dont le pourcentage de valeurs manquantes dépasse le seuil,
              triée par pourcentage décroissant
    """

    # Calcul du pourcentage de valeurs manquantes par colonne
    missing_percentages = (df.isnull().sum() / len(df)) * 100

    # Sélection des colonnes dépassant le seuil
    columns_above_threshold = missing_percentages[missing_percentages >= threshold]

    # classement par pourcentage décroissant
    columns_above_threshold = columns_above_threshold.sort_values(ascending=False)

    return  columns_above_threshold.index.to_list()

def quasi_constant_features(df:pd.DataFrame, threshold=0.01)->List:
    """
    detecte les colonnes avec une variance inférieure à un seuil donné : elle sont quasi contantes

    Args:
        df (pd.DataFrame): Le DataFrame à analyser
        threshold (float): Seuil de variance en dessous duquel on  considère une colonne comme quasi constante (par défaut: 0.01)
    Returns:
        list: Liste des noms de colonnes considérées comme quasi constantes
    """

    constant_cols = []
    # Traitement des variables numériques
    numerical_df = df.select_dtypes(include=[np.number, 'bool'])
    variances = numerical_df.var()
    constant_cols.extend(variances[variances < threshold].index.tolist())

    # Traitement des variables catégorielles (en regardant le ratio du mode)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    cat_low_variance = []
    for col in categorical_df.columns:
        # Calcul du ratio de la valeur la plus fréquente
        value_counts = categorical_df[col].value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] > 1 - threshold:
            cat_low_variance.append(col)

    # Combinaison des résultats
    constant_cols.extend(cat_low_variance)

    return constant_cols

def low_information_cols(df, constant_threshold=0.98, missing_threshold=0.01)->List:
    """
    Identifie les colonnes à faible information.(donc pas très utiles pour la modélisation).
    Cette fonction détecte les colonnes quasi-constantes et celles avec ont trop de veleurs manquantes

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        constant_threshold (float, optional): Seuil pour la quasi-constance.
                                              Par défaut, 0.98.
        missing_threshold (float, optional): pourcentage Seuil de valeurs maquante
                                         Par défaut, 0.01.

    Returns:
        list: Liste des noms de colonnes à faible information.
    """

    low_info_cols = []

    # Colonnes quasi-constantes
    low_info_cols.extend(quasi_constant_features(df, threshold=constant_threshold))
    # Colonnes avec trop de valeurs maquantes
    low_info_cols.extend(to_missing_values(df, threshold=missing_threshold))
    # Suppression des doublons
    low_info_cols = list(set(low_info_cols))

    return low_info_cols

def visualize_columns(df, columns=None , ncols:int=3, save=False, name:str= "visualize_columns")->None:
    """
    Affiche les graphiques pour les colonnes spécifiées du DataFrame , utiles pour verifier si les colonnes considérées sont non informatives
    """
    if columns is None:
        columns = df.columns.tolist()

    # Vérification que les colonnes existent dans le DataFrame
    columns = inter(columns, df.columns.tolist())

    if len(columns) == 0:
        print("Aucune colonne valide à visualiser.")
        return

    n = len(columns)
    nrows = (n // ncols) + (n%ncols)  # Ajustement du nombre de lignes

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  # Aplatir la matrice d'axes pour itération facile

    for i, col in enumerate(columns):
        ax = axes[i]
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution de {col}")
        else:
            sns.countplot(y=df[col], ax=ax)
            ax.set_title(f"Comptage de {col}")

        ax.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés
    for j in range(n + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join("Figures",f"{name}.png")) if save else None
    plt.show()

#=====================================================================

def apply_pca(df,columns, n_components=None,
              var_explained:float=0.7, verbose=True,
              handle_categorical:Literal['drop', 'ignore', 'encode'] = 'drop') -> Tuple[pd.DataFrame, PCA]:
    """
    Applique l'ACP sur les colonnes spécifiées du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        columns (list): Liste des colonnes à utiliser pour PCA.
        n_components (int, Optional): Nombre de composantes principales à conserver.
                si None utilise la moitié du nombre de colonnes
        var_explained (float): seuil de variance expliquée à atteindre pour déterminer le nombre de composantes.
        handle_categorical (str): comment gérer les colonnes catégorielles.
        - 'drop' : supprime les colonnes catégorielles
        - 'ignore' : ignore les colonnes catégorielles
        - 'encode' : encode les colonnes catégorielles en utilisant OneHotEncoder
        verbose (bool): Si True, affiche les résultats.

    Returns:
        Tuple[pd.DataFrame, PCA]: Un tuple contenant le dataframe avec les composantes principales et l'objet PCA entrainé
    """

    if not isinstance(columns, list):
        raise ValueError("Les colonnes doivent être fournies sous forme de liste.")


    # selection des données pour l'acp
    X = df[columns].copy()


    # gestion des colonnes catégorielles
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    if handle_categorical == 'drop':
        X = X.drop(columns=cat_cols)
        if verbose:
            print(f"Colonnes catégorielles supprimées : {cat_cols.tolist()}")
    elif handle_categorical == 'ignore':
        if verbose:
            print(f"Colonnes catégorielles ignorées : {cat_cols.tolist()}")
    elif handle_categorical == 'encode':
        # Encodage des colonnes catégorielles
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))  # Convertir en str pour éviter les erreurs

        if verbose and len(cat_cols)>0:
            print(f"Colonnes catégorielles encodées : {cat_cols.columns.tolist()}")

    # verification qu'il reste des colonnes numériques
    if X.shape[1] == 0:
        raise ValueError("Aucune colonne numérique restante après le traitement des colonnes catégorielles.")

    # standardisation des données
    scaler =RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # application de l'acp
    pca = PCA()
    pca.fit(X_scaled)

    # determination du nombre optimal de composantes
    if n_components is None :
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= var_explained) + 1
        if verbose:
            print(f"Nombre de composantes sélectionnées pour {var_explained*100} % de variance expliquée : {n_components}")

    # application de l'acp
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)],index=df.index)

    if verbose:
        print(f"Composantes principales créées : {pca_df.columns.tolist()}")

    # Suppression des colonnes originales
    df = df.drop(columns=columns)

    # Ajout des composantes principales au DataFrame
    df = pd.concat([df, pca_df], axis=1)

    return df , pca

def interpret_pca(pca, columns,top_n:int=5):
    """
    donne une interprétation des composantes principales en affichant les poids des variables originales

    Args:
        pca (PCA): L'objet PCA entraîné.
        columns (list): Liste des noms de colonnes originales.
        top_n (int): Nombre de variables les plus importantes à afficher pour chaque composante."""

    for i , component in enumerate(pca.components_):
        print(f"\nComposante principale {i+1}:")
        component_weights = pd.Series(component, index=columns)
        # Tri des poids par valeur absolue
        top_vars = component_weights.abs().nlargest(top_n)
        # Affichage des variables les plus importantes
        print(top_vars)


def all_columns(df, res=True):
    all_col = df.columns.tolist()
    print(f" il y'a {len(all_col)} colonnes dans le dataframe")
    print(f"ces colonnes sont : \n{df.columns}")
    sep()
    if res:
        return all_col
    else:
        return None


def display_correlation_matrix(df, table=False, target=None, threshold=0.5  , save=True , name:str='corr_matrix1'):
    """
    Affiche la matrice de corrélation entre les colonnes numériques du DataFrame.
    """
    t_in = (target in df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        raise ValueError("Aucune colonne numérique trouvée dans le DataFrame.")

    if (target is not None) and t_in:
        num_cols = [target] + [col for col in num_cols if col != target]

    corr_matrix = df.copy()[num_cols].corr()

    # Correction de la création du masque triangulaire
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Correction de la ligne problématique
    filtered_corr = upper[
        (upper.abs() > threshold) &
        (upper != 1.0)
    ].dropna(how='all', axis=0).dropna(axis=1, how='all')

    if table:
        display(filtered_corr)
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(upper, annot=True, cmap="RdBu")
        plt.savefig(os.path.join("Figures",f"{name}.png")) if save else None
        plt.show()


def get_target_correlations(df, target, threshold=0.5, verbose=True):
    """
    Identifie les colonnes fortement corrélées avec la variable cible.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée
        target (str): Nom de la colonne cible
        threshold (float): Seuil de corrélation (défaut: 0.5)
        verbose (bool): Si True, affiche les résultats

    Returns:
        list: Liste des colonnes fortement corrélées avec la cible
    """
    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' n'existe pas dans le DataFrame")

    # Calcul des corrélations avec la cible
    correlations = df.corr()[target].sort_values(ascending=False)

    # Filtrage des corrélations significatives (en excluant la cible elle-même)
    strong_corr = correlations[
        (correlations.abs() >= threshold) &
        (correlations.index != target)
    ]

    if verbose:
        print(f"\nColonnes fortement corrélées avec {target} (|corr| >= {threshold}):")
        print(strong_corr)

    return strong_corr.index.tolist()

def get_correlated_pairs(df, threshold=0.5, verbose=True):
    """
    Identifie les paires de colonnes fortement corrélées entre elles.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée
        threshold (float): Seuil de corrélation (défaut: 0.5)
        verbose (bool): Si True, affiche les résultats

    Returns:
        list: Liste de tuples (colonne1, colonne2, correlation)
    """

    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("Aucune colonne numérique trouvée dans le DataFrame")

    # Calcul de la matrice de corrélation
    corr_matrix = df[numeric_cols].corr()

    # Création d'une liste de paires corrélées
    correlated_pairs = []

    # Parcours de la matrice triangulaire supérieure
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                col1 = corr_matrix.index[i]
                col2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                correlated_pairs.append((col1, col2, correlation))

    # Tri par corrélation absolue décroissante
    correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if verbose:
        print(f"\nPaires de colonnes fortement corrélées (|corr| >= {threshold}):")
        for col1, col2, corr in correlated_pairs:
            print(f"{col1:<20} - {col2:<20}: {corr:.3f}")
        sep()

    return correlated_pairs

def select_best_features(df, target, threshold=0.5):
    """
    Sélectionne les meilleures features en éliminant les variables corrélées.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée
        target (str): Nom de la colonne cible
        threshold (float): Seuil de corrélation

    Returns:
        list: Liste des colonnes à conserver
    """
    num_col = df.select_dtypes(include=[np.number]).columns
    if target not in num_col:
        raise ValueError(f"La colonne cible '{target}' n'existe pas dans le DataFrame")
    if len(num_col) == 0:
        raise ValueError("Aucune colonne numérique trouvée dans le DataFrame")

    dfs = df[num_col].copy()

    # Obtention des corrélations avec la cible
    target_corr = pd.Series({
        col: abs(dfs[col].corr(dfs[target]))
        for col in dfs.columns if col != target
    })

    # Obtention des paires corrélées
    correlated_pairs = get_correlated_pairs(df, threshold, verbose=False)

    # Colonnes à éliminer
    to_drop = set()

    # Pour chaque paire corrélée, on garde celle qui a la plus forte corrélation avec la cible
    for col1, col2, _ in correlated_pairs:
        if col1 != target and col2 != target:
            if target_corr[col1] < target_corr[col2]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)

    # Liste finale des colonnes à conserver
    columns_to_keep = [col for col in df.columns if col not in to_drop and col != target]

    print(f"\nColonnes  éliminées: {sorted(to_drop)}")
    print(f"Colonnes  conservées: {sorted(columns_to_keep)}")

    return columns_to_keep , list(to_drop)