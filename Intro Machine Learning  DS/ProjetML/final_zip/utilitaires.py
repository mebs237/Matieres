"""
    modules de fonctions utilitaires utiliser pour l'exploration et le prétraitement des données
"""


from typing import List, Literal, Tuple , Optional
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler , LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from os.path import join
from pandas import DataFrame , Series , concat
from numpy import (triu , ones , zeros , nan , number , sqrt , cumsum , argmax)
from pandas.api.types import is_numeric_dtype
from seaborn import heatmap , histplot , countplot
from tqdm import tqdm



#=====================================================================

# @title Constantes globales
MISSING_THRESHOLD = 0.5 # @param {"type":"float"}
ENTROPY_THRESHOLD = 0.1 #@param {"type":"float"}
VARIANCE_THRESHOLD = 0.01 #@param{"type":"float"}
NMI_THRESHOLD = 0.8 #@param{"type":"float"}
RANDOM_STATE = 42 # @param {"type":"integer"}


#=====================================================================

def sep(lg:int=90)->str:
    """Affiche une ligne de séparation"""
    print("\n" + "-"*lg + "\n")

def sub(l1:List,l2:List)->List:
    """Retourne la liste des éléments de l1 qui ne sont pas dans l2"""
    return list(set(l1) - set(l2))

def inter(l1:List,l2:List)->List:
    """Retourne la liste des éléments communs entre l1 et l2"""
    return list(set(l1) & set(l2))

def all_columns(df:DataFrame, res:bool=True):
    """ resumé sur les colonnes """
    all_col = df.columns.tolist()
    print(f" il y'a {len(all_col)} colonnes dans le dataframe")
    print(f"ces colonnes sont : \n{df.columns}")
    sep()
    if res:
        return all_col
    else:
        return None


#=====================================================================
# Fonctions de nettoyage des données
#=====================================================================

def discretize_column(col:Series,
                      n_bins:int=10,
                      strategy: Literal['quantile','uniform','kmeans'] = 'quantile')-> Series:
    """
    Discrétise une colonne numérique en utilisant KBinsDiscretizer pour créer des bins uniformes.

    Args:
        col (Series): Colonne à discrétiser.
        strategy (str):  strategie de discretisation à utiliser
        n_bins (int): Nombre de bins à créer (par défaut: 10)

    Returns:
        pd.Serie: Colonne discrétisée.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    values = discretizer.fit_transform(col.values.reshape(-1, 1)).flatten()
    return Series(values, index=col.index)


def low_entropy_cols(df:DataFrame,
                     threshold:float=ENTROPY_THRESHOLD,
                     nbins:int=10,
                     strategy:Literal['quantile','uniform','kmeans'] ='quantile')->List[str]:
    """
    Identifie les colonnes avec une entropie inférieure à un seuil donné, indiquant qu'elles sont peu informatives.

    Args:
        df (DataFrame): Le DataFrame à analyser.
        threshold (float): Seuil d'entropie en dessous duquel une colonne est considérée comme peu informative (par défaut: 0.1).
        nbins (int): Nombre de bins pour le calcul de l'entropie des colonnes numériques (par défaut: 10).
    Returns:
        list: Liste des noms de colonnes considérées comme peu informatives.
    """

    data = df.copy()
    if data.empty :
        print("Le DataFrame est vide ")
        return []

    low_entropy = []

    for col in data.columns:
        serie = data[col].dropna()
        # si la colonne est numérique , on la discretise
        if is_numeric_dtype(serie):
            if strategy !='quantile':
                serie = Series(RobustScaler().fit_transform(serie.values.reshape(-1, 1)).flatten(),index = serie.index)
            serie = discretize_column(serie,n_bins=nbins,strategy=strategy)

        probs = serie.value_counts(normalize=True)
        col_entropy = entropy(probs,base=2)
        if col_entropy < threshold:
            low_entropy.append(col)

    return low_entropy

def to_missing_values(df:DataFrame, threshold:int=MISSING_THRESHOLD)->List[str]:
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
    missing_ratio = df.isnull().sum() / len(df)

    # Sélection des colonnes dépassant le seuil
    columns_above_threshold = missing_ratio[missing_ratio >= threshold]

    # classement par pourcentage décroissant
    columns_above_threshold = columns_above_threshold.sort_values(ascending=False)

    return  columns_above_threshold.index.to_list()

def quasi_constant_cols(df:DataFrame, threshold=VARIANCE_THRESHOLD)->List[str]:
    """
    detecte les colonnes avec une variance inférieure à un seuil donné : elle sont quasi contantes

    Args:
        df (DataFrame): Le DataFrame à analyser
        threshold (float): Seuil de variance en dessous duquel on  considère une colonne comme quasi constante (par défaut: 0.01)
    Returns:
        list: Liste des noms de colonnes considérées comme quasi constantes
    """

    data = df.copy()
    if data.empty :
        print("Le DataFrame est vide après suppression des valeurs manquantes.")
        return []

    constant_cols = []
    # Traitement des variables numériques avec scaling
    numerical_df = data.select_dtypes(include=[number])
    if not numerical_df.empty:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(numerical_df)
        variances = DataFrame(scaled,columns=numerical_df.columns).var()
        constant_cols.extend(variances[variances < threshold].index.tolist())

    # Traitement des variables catégorielles (en regardant le ratio du mode)
    categorical_df = data.select_dtypes(include=['object', 'category'])
    cat_low_variance = []
    for col in categorical_df.columns:
        # Calcul du ratio de la valeur la plus fréquente
        value_counts = categorical_df[col].dropna().value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] > 1 - threshold:
            cat_low_variance.append(col)

    # Combinaison des résultats
    constant_cols.extend(cat_low_variance)

    return constant_cols

def low_information_cols(df:DataFrame,
                         constant_threshold:float=VARIANCE_THRESHOLD,
                         missing_threshold:float= MISSING_THRESHOLD,
                         entropy_threshold:float=ENTROPY_THRESHOLD,
                         n_bins:int=10,
                         strategy:str='quantile')->List[str]:
    """
    Identifie les colonnes à faible information.(donc pas très utiles pour la modélisation).
    Elle détecte les colonnes :
    - quasi-constantes (donc variance faible)
    - avec trop de valeurs manquantes
    - avec une entropie faible

    Args:
        df (DataFrame): DataFrame d'entrée.
        constant_threshold (float, optional): Seuil pour la quasi-constance.
                                              Par défaut, 0.98.
        missing_threshold (float, optional): pourcentage Seuil de valeurs maquante
                                         Par défaut, 0.01.
        entropy_threshold (float, optional): Seuil d'entropie pour les colonnes catégorielles.

    Returns:
        List[str]: Liste des noms de colonnes à faible information.
    """

    low_info_cols = set()

    # Colonnes avec trop de valeurs manquantes
    low_info_cols.update(to_missing_values(df, threshold=missing_threshold))

    # Colonnes quasi-constantes
    low_info_cols.update(quasi_constant_cols(df, threshold=constant_threshold))

    # colonnes avec une entropie faible
    low_info_cols.update(low_entropy_cols(df, threshold=entropy_threshold,nbins=n_bins,strategy=strategy))


    return list(low_info_cols)

combi = namedtuple('combi', ['var_thresh', 'ent_thresh', 'mis_thresh', 'n_bins', 'strategy', 'nbcol', 'to_drop', 'score'])

def new_score_combination(var_thresh:float, ent_thresh:float,
                          mis_thresh:float, n_bins:int,
                          strategy: str, nbcol: int,
                          global_info: dict, total_cols: int) -> float:

    # 1. Score basé sur la conformité aux distributions globales
    # On pénalise si les seuils sont trop permissifs (couvrent une petite proportion de colonnes)
    # Les poids sont basés sur l'idée que chaque type de seuil a la même importance
    # 📝 Remarque : vous pourriez ajuster ces poids (30, 30, 30)
    var_coverage = (global_info["var_percentiles"] < var_thresh).mean()
    ent_coverage = (global_info["ent_percentiles"] < ent_thresh).mean()
    mis_coverage = (global_info["missing_percentiles"] > mis_thresh).mean()

    conformity_score = (var_coverage + ent_coverage + mis_coverage) * 10

    # 2. Score basé sur l'impact (nombre de colonnes supprimées)
    # Le poids de `nbcol` est fort, car c'est l'objectif principal
    impact_score = (nbcol / total_cols) * 100

    # 3. Bonus/Malus sur les autres paramètres
    complexity_bonus = 0
    complexity_bonus -= n_bins / 20 # Pénalisation de la complexité
    complexity_bonus += {"quantile": 2, "uniform": 1, "kmeans": -1}.get(strategy, 0)

    # 4. Combinaison du score
    # On donne un poids important à `nbcol` et à la conformité
    final_score = impact_score * 0.7 + conformity_score * 0.3 + complexity_bonus

    return round(final_score, 2)


def score_combination(var_thresh, ent_thresh, mis_thresh, n_bins, strategy, nbcol):
    """
    Calcule un score combiné pour prioriser les combinaisons :
    - + points pour plus de colonnes supprimées
    - + points pour des seuils plus stricts (petits)
    - bonus ou malus selon la stratégie
    """
    score = 0
    score += nbcol * 10                      # priorité forte : colonnes supprimées
    score -= var_thresh * 100                # petite variance → meilleure détection
    score -= ent_thresh * 100                # faible entropie → plus sévère
    score -= mis_thresh * 50                 # moins tolérant aux NaNs
    score -= n_bins                          # moins de bins = plus simple
    score += {'quantile': 0, 'uniform': 1, 'kmeans': 2}.get(strategy,0)  # stratégie plus "lourde" pénalisée
    return round(score, 2)

def tune_thresholds(df:DataFrame,
                    var_range:List=[0.001, 0.01, 0.05],
                    ent_range:List=[0.05, 0.1, 0.2],
                    mis_range:List=[0.3, 0.5, 0.7],
                    strategies:List[str]=['quantile','uniform','kmeans'],
                    bins_range:List[int]=[5,10,20],
                    verbose:bool=True,restrict:bool=False)->List[combi]:
    """
    Teste différentes combinaisons de seuils pour identifier les colonnes à faible information."""
    results = []
    nbmax = 0
    nbiter = 3**5
    nbiter = len(var_range) * len(ent_range) * len(mis_range) * len(bins_range) * len(strategies)

    with tqdm(total = nbiter,desc="progression") as pbar:
        for var_thresh in var_range :
            for ent_thresh in ent_range:
                for mis_thresh in mis_range:
                    for n_bins in bins_range:
                        for strategy in strategies:
                            to_drop = low_information_cols(df, constant_threshold=var_thresh, entropy_threshold=ent_thresh,
                            missing_threshold=mis_thresh, n_bins=n_bins, strategy=strategy)
                            nbcol = len(to_drop)
                            score = score_combination(var_thresh, ent_thresh, mis_thresh, n_bins, strategy, nbcol)
                            if nbcol > nbmax:
                                nbmax = nbcol
                            results.append(
                                combi(var_thresh, ent_thresh,
                                       mis_thresh,n_bins,
                                       strategy, nbcol,
                                       to_drop,score))
                            pbar.update(1)
    best_results = sorted(results, key=lambda x: (x.nbcol , x.score),reverse=True)
    if restrict:
        best_results = list(filter(lambda x: x.nbcol == nbmax, best_results))


    if verbose:
        for r in best_results:
            print(f"score:{r.score} | var:{r.var_thresh} | ent:{r.ent_thresh} | mis:{r.mis_thresh} | n_bins:{r.n_bins} | strategy:{r.strategy} => {r.nbcol} colonnes")
    return best_results

def visualize_columns(df, columns=None ,
                      ncols:int=3, save=False,
                      name:str= "visualize_columns")->None:
    """
    Affiche les graphiques pour verifier si les colonnes considérées sont non informatives
    """
    df_used = df.copy().dropna()

    if columns is None:
        columns = df_used.columns.tolist()

    # Vérification que les colonnes existent dans le DataFrame
    columns = inter(columns, df_used.columns.tolist())

    if len(columns) == 0:
        print("Aucune colonne valide à visualiser.")
        return

    n = len(columns)
    nrows = (n // ncols) + (n%ncols>0)  # Ajustement du nombre de lignes

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols * 4), int(nrows * 5)))
    axes = axes.flatten()  # Aplatir la matrice d'axes pour itération facile

    for i, col in enumerate(columns):
        ax = axes[i]
        if is_numeric_dtype(df_used[col]):
            histplot(data=df_used,x=col, kde=True, ax=ax)
            ax.set_title(f"Distribution de {col}")
        else:
            countplot(data=df_used , x=col, ax=ax)
            ax.set_title(f"Comptage de {col}")

        ax.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés
    for j in range(n , len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save:
        plt.savefig(join("Figures",f"{name}.png"))
    plt.show()

#=====================================================================
# Fonctions de regroupement par ACP
#=====================================================================

def apply_pca(df:DataFrame,columns:List[str],
              n_components:Optional[int]=None,
              var_explained:float=0.7, verbose=True,
              handle_categorical:Literal['drop', 'ignore', 'encode'] = 'drop') -> Tuple[DataFrame, PCA]:
    """
    Applique l'ACP sur les colonnes spécifiées du DataFrame.

    Args:
        df (DataFrame): Le DataFrame d'entrée.
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
        Tuple[DataFrame, PCA]: Un tuple contenant le dataframe avec les composantes principales et l'objet PCA entrainé
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
        cumulative_variance = cumsum(pca.explained_variance_ratio_)
        n_components = argmax(cumulative_variance >= var_explained) + 1
        if verbose:
            print(f"Nombre de composantes sélectionnées pour {var_explained*100} % de variance expliquée : {n_components}")

    # application de l'acp
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)],index=df.index)

    if verbose:
        print(f"Composantes principales créées : {pca_df.columns.tolist()}")

    # Suppression des colonnes originales
    df = df.drop(columns=columns)

    # Ajout des composantes principales au DataFrame
    df = concat([df, pca_df], axis=1)

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
        component_weights = Series(component, index=columns)
        # Tri des poids par valeur absolue
        top_vars = component_weights.abs().nlargest(top_n)
        # Affichage des variables les plus importantes
        print(top_vars)


#=====================================================================
# Fonctions d'éliminations des colonnes redondantes
#=====================================================================


def mutual_info_norm(x:Series,
                     y:Series,
                     n_bins:int = 10,
                     strategy: str = 'quantile')->float:
    """
    Calcule la mutual_information normalisée entre deux variables d'un dataframe

    Args:
        x (Series): Première variable
        y (Series): Deuxième variable
        n_bins (int, optional): Nombre de bins pour la discrétisation des variables numériques (par défaut: 10)
        strategy (str, optional): Stratégie de discrétisation ('quantile' ou 'uniforme', par défaut: 'quantile')
    """

    if x.empty or y.empty:
        return 0.0

    # Discretisation des variables numériques si necessaires
    if is_numeric_dtype(x):
        x = discretize_column(x, n_bins=n_bins, strategy=strategy)
    else:
        x= x.astype(str
                    )
    if is_numeric_dtype(y):
        y = discretize_column(y, n_bins=n_bins, strategy=strategy)
    else:
        y = y.astype(str)

    try :
        # calcul de la mutual_information
        mi = mutual_info_score(x,y)

        # calcul des entropies
        h_x = entropy(x.value_counts(normalize=True), base=2)
        h_y = entropy(y.value_counts(normalize=True), base=2)

        # eviter la division par zero (si une entropie est nulle )
        if h_x ==0 or h_y ==0:
            return 0.0

        # normalisation
        nmi = mi /sqrt(h_x*h_y)
        return nmi
    except Exception as e:
        print(f"Erreur lors du calcul de la mutual_information entre {x.name} et {y.name}: {e}")
        return 0.0


def mutual_info_matrix(df:DataFrame,
                      target:Optional[str]=None):
    """
    Matrice de mutual_information normalisée entre les colonnes du Dataframe

    Args:
        df (DataFrame): Le DataFrame d'entrée
        target (str, optional): Nom de la colonne cible. Si None, utilise toutes les colonnes.

    Returns:
        DataFrame: Matrice de mutual_information entre les colonnes du DataFrame
    """

    df = df.copy()
    cols = df.columns.tolist()
    if target and target in cols:
        cols.remove(target)
        cols = [target] + cols
    n = len(cols)
    matrix = DataFrame(zeros((n,n)) , index=cols, columns=cols, dtype=float)

    for i, c1 in enumerate(cols):
        for _ , c2 in enumerate(cols[i:], start=i):
            if c1 == c2:
                matrix.loc[c1, c2] = 1.0
            else:
                sub_df = df[[c1, c2]].dropna()
                if sub_df.empty:
                    mi = nan
                else:
                    mi = mutual_info_norm(sub_df[c1], sub_df[c2])
                matrix.loc[c1, c2] = mi
                matrix.loc[c2, c1] = mi
    return matrix


def display_mutual_info_matrix(df:Optional[DataFrame]=None,
                               target:Optional[str]=None,
                               matrix:Optional[DataFrame]=None,
                               threshold=0.0,
                               save:bool=False,
                               name:str='mutual_info_matrix'):
    """
    Affiche la matrice triangulaire supérieure de mutual_information entre les colonnes du DataFrame

    Args:
        df (DataFrame): Le DataFrame d'entrée , peut être  None si matrix est déjà calculée
        target (str, optional): Nom de la colonne cible. Si None, utilise toutes les colonnes.
        matrix (DataFrame, optional): Matrice de mutual_information pré-calculée. Si None, elle sera calculée à partir du DataFrame.
        threshold (float, optional): Seuil pour filtrer les valeurs de la matrice (par défaut: 0.0 pour afficher toutes les valeurs )
        save (bool, optional): Si True, enregistre le graphique dans un fichier (par défaut: True)
        name (str, optional): Nom du fichier à enregistrer (par défaut: 'mutual_info_matrix')
    """
    if matrix is None:
        if df is None:
            raise ValueError("Veuillez fournir un DataFrame ou une matrice de mutual_information.")
        if df.empty:
            print("Le DataFrame est vide.")
            return
        matrix = mutual_info_matrix(df, target = target)

    if matrix.shape[0]!= matrix.shape[1]:
        raise ValueError("La matrice de mutual_information doit être carrée.")

    # Filtrage des valeurs au-dessus du seuil
    mask = triu(ones(matrix.shape), k=1).astype(bool)
    filtered_matrix = matrix.where(mask)
    filtered_matrix = filtered_matrix[filtered_matrix>=threshold]

    plt.figure(figsize=(12, 10))
    heatmap(filtered_matrix,
                mask=~mask,
                annot=True,
                cmap="RdBu",
                fmt=".2f",
                vmin=0, vmax=1)
    plt.title(f"matrice de mutual_information normalisée {name}")
    plt.tight_layout()
    if save:
        plt.savefig(join("Figures",f"{name}.png"))
    plt.show()


def get_dependant_pairs(df:DataFrame=None,
                        threshold=NMI_THRESHOLD,
                        matrix=None,
                        verbose:bool=False
                        )-> List[Tuple[str, str]]:
    """
    Identifie les paires de colonnes dépendantes dans le DataFrame en utilisant la matrice de mutual_information.

    Args:
        df (DataFrame): Le DataFrame d'entrée
        threshold (float): Seuil pour filtrer les valeurs de la matrice (par défaut: 0.0 pour afficher toutes les valeurs )
        matrix (DataFrame, optional): Matrice de mutual_information pré-calculée. Si None, elle sera calculée à partir du DataFrame.
        verbose (bool): Si True, affiche les résultats

    Returns:
        list: Liste de tuples contenant les noms des colonnes dépendantes
    """
    dependant_pairs = []

    if matrix is None:
        if df is None :
            raise ValueError("Veuillez fournir un DataFrame ou une matrice de mutual_information.")
        if df.empty:
            print("Le DataFrame est vide.")
            return dependant_pairs
        matrix = mutual_info_matrix(df)

    if matrix.shape[0]!= matrix.shape[1]:
        raise ValueError("La matrice de mutual_information doit être carrée.")

    n = matrix.shape[0]

    if n == 0:
        print("Le DataFrame ne contient aucune colonne.")
        return dependant_pairs

    for i in range(n):
        for j in range(i + 1, n):
            col1 = matrix.columns[i]
            col2 = matrix.columns[j]
            if matrix.iloc[i, j] >= threshold:
                dependant_pairs.append((col1, col2 , matrix.iloc[i, j]))
    dependant_pairs = sorted(dependant_pairs, key=lambda x: x[2], reverse=True)
    if verbose:
        print(f"\n {len(dependant_pairs)} Paires de colonnes dépendantes (mutual_info >= {threshold:.2f}):")
        for col1, col2 , score in dependant_pairs:
            print(f"{col1:<20} | {col2:<20} : {score:.3f} ")

    return sorted(dependant_pairs , key=lambda x: x[2],reverse=True)

def display_pairs(liste:List):
    """
    Affiche les paires de colonnes dépendantes d'une liste de tuples
    """
    if not liste:
        print("Aucune paire de colonnes dépendantes trouvée.")
        return

    print(f"\n {len(liste)} Paires de colonnes dépendantes:")
    for col1, col2, score in liste:
        print(f"{col1:<21} | {col2:<20} : nmi = {score:.3f} ")
def remove_redundant_cols(df:DataFrame=None,
                          target:str=None,
                          threshold:float=NMI_THRESHOLD,
                          matrix:DataFrame=None,
                          verbose:bool=False)->Tuple[List[str], List[str]]:
    """
    Supprime les variable redondante en conservant celle avec le plus haut score de mutual_information avec la cible

    Parameters
    ----------
    df : DataFrame
        dataframe d'entrée
    target : str , optional
        nom de la colonne cible , si ``None`` , on conserve le premiers élément de chaque paire dépendante
    threshold : float, optional
        seuil de forte dépendance , par default 0.7
    matrix : DataFrame, optional
        matrice d'information mutuelle, si None, elle sera calculée à partir de df
    verbose : bool, optional
        si ``True`` affiche les décisions prises durant l'analyse

    Returns
    -------
    Tuple[List[str], List[str]]
       couple , de la liste de colonnes à conserver et la liste des colonnes suprimées durand l'analyse
    """

    if matrix is None:
        if df is None :
            raise ValueError("Veuillez fournir un DataFrame ou une matrice de mutual_information.")
        if df.empty:
            print("Le DataFrame est vide.")
            return [] , []
        matrix = mutual_info_matrix(df)

    if matrix.shape[0]!= matrix.shape[1]:
        raise ValueError("La matrice de mutual_information doit être carrée.")

    dependant_pairs = get_dependant_pairs(df=df, threshold=threshold, matrix=matrix, verbose=False)
    if len(dependant_pairs) == 0:
        print("Aucune paire de colonnes dépendantes trouvée.")
        return matrix.columns.tolist(), []

    to_drop = set()
    alea = target is None

    for var1 , var2 , _ in dependant_pairs:
        if alea :
            to_drop.add(var2)
            if verbose:
                print(f"({var1:<20} | {var2:<20}) ---x {var2}")
        else:
            if var1 != target or var2 != target:
                nmi_var1_target = matrix.loc[var1,target]
                nmi_var2_target = matrix.loc[var2,target]

                if nmi_var1_target >= nmi_var2_target:
                    to_drop.add(var2)
                    if verbose:
                        print(f"({var1:<20} | {var2:^22}) : ---x {var2}")
                else:
                    to_drop.add(var1)
                    if verbose:
                        print(f"({var1:<20} | {var2:<20}) ---x {var1}")

    return list(set(matrix.columns) - to_drop), list(to_drop)

def display_relative_importance(df:DataFrame=None,
                                matrix:DataFrame=None,
                                colums:List[str]=None,
                                target:str=None):

    if matrix is None:
        if df is None :
            raise ValueError("Veuillez fournir un DataFrame ou une matrice de mutual_information.")
        if df.empty:
            print("Le DataFrame est vide.")
            return [] , []
        matrix = mutual_info_matrix(df)

    if target is None or target not in matrix.columns:
        if colums is None or len(colums) == 0:
            target = matrix.columns[0]
        target = matrix.columns[0]

    valids_cols = sub(matrix.columns.tolist(), [target])

    scores = matrix.loc[valids_cols,target].sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    scores.plot(kind='barh', color='skyblue')
    plt.title(f"importance relative des variables par rapport à {target}")
    plt.ylabel("Mutual Information")
    plt.xlabel("Variables")
    plt.xticks(rotation=45)
    plt.grid(axis='y',linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()