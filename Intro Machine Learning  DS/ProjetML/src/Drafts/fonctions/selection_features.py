from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import warnings
from typing import Dict, List, Union, Tuple, Any, Optional

# Définition d'une constante pour la reproductibilité
RANDOM_STATE = 42

def get_preprocessor(X):
    """
    Crée un pipeline de prétraitement pour les variables numériques et catégorielles.

    Args:
        X (pd.DataFrame): DataFrame contenant les données à prétraiter

    Returns:
        ColumnTransformer: Transformateur pour le prétraitement
    """
    X = X.copy()

    # Identification des types de colonnes
    num_col = X.select_dtypes(include=['int64', 'float64']).columns
    cat_col = X.select_dtypes(include=['object', 'category', 'bool']).columns
    date_col = X.select_dtypes(include=['datetime64']).columns

    transformers = []

    # Transformation des variables numériques (si présentes)
    if len(num_col) > 0:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        transformers.append(('num', num_transformer, num_col))

    # Transformation des variables catégorielles (si présentes)
    if len(cat_col) > 0:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_transformer, cat_col))

    # Transformation des variables date (si présentes)
    if len(date_col) > 0:
        date_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('extract_features', FunctionTransformer(lambda x: pd.DataFrame({
                'year': x.dt.year,
                'month': x.dt.month,
                'day': x.dt.day,
                'dayofweek': x.dt.dayofweek
            })))
        ])
        transformers.append(('date', date_transformer, date_col))

    # Création du transformateur de colonnes
    if not transformers:
        warnings.warn("Aucune colonne valide trouvée pour le prétraitement.")
        # Transformer vide pour éviter les erreurs
        return ColumnTransformer(transformers=[('placeholder', 'passthrough', [])])

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def identify_low_variance_features(df, threshold=0.01, display=True):
    """
    Identifie les caractéristiques à faible variance.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        threshold (float): Seuil de variance en dessous duquel une caractéristique est considérée à faible variance
        display (bool): Si True, affiche les informations sur les colonnes identifiées

    Returns:
        list: Liste des colonnes à faible variance (à supprimer)
    """
    # Traitement des variables numériques
    numerical_df = df.select_dtypes(include=[np.number, 'bool'])
    variances = numerical_df.var()
    low_variance_cols = variances[variances < threshold].index.tolist()

    # Traitement des variables catégorielles (en regardant le ratio du mode)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    cat_low_variance = []

    for col in categorical_df.columns:
        # Calcul du ratio de la valeur la plus fréquente
        value_counts = categorical_df[col].value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] > 1 - threshold:
            cat_low_variance.append(col)

    # Combinaison des résultats
    all_low_variance = low_variance_cols + cat_low_variance

    if display:
        print(f"Colonnes numériques à faible variance : {low_variance_cols}")
        print(f"Colonnes catégorielles à faible variance : {cat_low_variance}")
        print(f"Total des colonnes à faible variance : {len(all_low_variance)}")

    return all_low_variance

def identify_correlated_features(df, target=None, threshold=0.8, disp_save_corr=False):
    """
    Identifie les caractéristiques fortement corrélées entre elles.
    Si target est fourni, conserve celle qui a la plus forte corrélation avec la cible.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        target (str, optional): Nom de la variable cible
        threshold (float): Seuil de corrélation au-dessus duquel les caractéristiques sont considérées comme fortement corrélées
        disp_save_corr (bool): Si True, affiche et sauvegarde la matrice de corrélation

    Returns:
        list: Liste des colonnes à supprimer
    """
    try:
        data = df.copy()
        num_col = data.select_dtypes(include=[np.number]).columns

        # Si aucune colonne numérique, retourner liste vide
        if len(num_col) == 0:
            return []

        # Exclure la cible des calculs de corrélation si elle est fournie
        df_numeric = data[num_col]
        if target in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=[target])

        # S'il ne reste plus de colonnes, retourner liste vide
        if df_numeric.shape[1] <= 1:
            return []

        # Calcul de la matrice de corrélation
        corr_mat = df_numeric.corr()
        corr_matrix = corr_mat.abs()

        # Calcul de la corrélation avec la cible (si fournie)
        target_corr = None
        if target is not None and target in data.columns:
            if pd.api.types.is_numeric_dtype(data[target]):
                target_corr = df_numeric.corrwith(data[target]).abs()

        # Identification des paires fortement corrélées
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()

        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > threshold:
                    # Décision basée sur la corrélation avec la cible (si disponible)
                    if target_corr is not None:
                        if target_corr[col] > target_corr[row]:
                            to_drop.add(row)
                        else:
                            to_drop.add(col)
                    else:
                        # Sans cible, on garde arbitrairement la première variable
                        to_drop.add(col)

        # Affichage de la matrice de corrélation
        if disp_save_corr:
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title("Matrice de corrélation")
            plt.tight_layout()
            plt.savefig("corr_matrix.png")
            plt.show()

        return list(to_drop)

    except Exception as e:
        print(f"Erreur dans identify_correlated_features: {e}")
        return []

def get_feature_names_from_preprocessor(preprocessor, X):
    """
    Extrait les noms des caractéristiques après prétraitement.

    Args:
        preprocessor (ColumnTransformer): Transformateur ajusté
        X (pd.DataFrame): DataFrame original

    Returns:
        list: Liste des noms des caractéristiques après prétraitement
    """
    try:
        # Vérifier si le transformateur est vide
        if not preprocessor.transformers_:
            return []

        # Tentative d'utiliser get_feature_names_out si disponible (scikit-learn >= 1.0)
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out())

        # Méthode alternative pour les versions plus anciennes
        feature_names = []

        # Traitement des colonnes numériques
        if 'num' in preprocessor.named_transformers_:
            num_col = X.select_dtypes(include=['int64', 'float64']).columns
            feature_names.extend(num_col)

        # Traitement des colonnes catégorielles
        if 'cat' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
            if 'onehot' in preprocessor.named_transformers_['cat'].named_steps:
                cat_col = X.select_dtypes(include=['object', 'category', 'bool']).columns
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_col)
                feature_names.extend(cat_features)

        # Traitement des colonnes date
        if 'date' in preprocessor.named_transformers_:
            date_col = X.select_dtypes(include=['datetime64']).columns
            for col in date_col:
                feature_names.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])

        return feature_names

    except Exception as e:
        print(f"Erreur lors de l'extraction des noms de caractéristiques: {e}")
        # Fallback: utiliser les indices comme noms
        return [f"feature_{i}" for i in range(preprocessor.transform(X.head(1)).shape[1])]


def select_important_features(X, y):
    # Prétraitement des données
    X_copy = X.copy()
    for col in X_copy.select_dtypes(include=['object']).columns:
        if X_copy[col].apply(type).nunique() > 1:
            X_copy[col] = X_copy[col].astype(str)

    # Application du preprocesseur
    preprocessor = get_preprocessor(X_copy)
    X_processed = preprocessor.fit_transform(X_copy)
    feature_names = get_feature_names(preprocessor, X_processed)

    # LassoCV - travaille directement avec la matrice sparse
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE)
    lasso_cv.fit(X_processed, y)
    lasso_coef = pd.Series(lasso_cv.coef_, index=feature_names)
    lasso_top = lasso_coef[lasso_coef != 0].index.tolist()

    # RandomForest - utilise la matrice sparse
    rf = RandomForestRegressor(random_state=RANDOM_STATE, sparse_output=True)
    rf.fit(X_processed, y)
    rf_importance = pd.Series(rf.feature_importances_, index=feature_names)
    rf_top = rf_importance.sort_values(ascending=False).head(20).index.tolist()

    # XGBoost - configuration pour matrices sparse
    xgb = XGBRegressor(
        random_state=RANDOM_STATE,
        enable_categorical=True,
        tree_method='hist'  # Plus efficace pour les grandes matrices
    )
    xgb.fit(X_processed, y)
    xgb_importance = pd.Series(xgb.feature_importances_, index=feature_names)
    xgb_top = xgb_importance.sort_values(ascending=False).head(20).index.tolist()

    # Consolidation des features importantes
    important_features = list(set(lasso_top + rf_top + xgb_top))
    return important_features



def backward_stepwise_selection(model, X, y, preprocessor=None, initial_features=None, min_features=None, rtol=0.001, cv=5):
    """
    Sélection de caractéristiques par élimination progressive.

    Args:
        model: Modèle à utiliser pour l'évaluation
        X (pd.DataFrame): DataFrame des caractéristiques
        y (pd.Series): Variable cible
        preprocessor: Préprocesseur à utiliser
        initial_features (list): Liste initiale des caractéristiques à considérer
        min_features (int): Nombre minimum de caractéristiques à conserver
        rtol (float): Tolérance relative pour l'amélioration de la performance
        cv (int): Nombre de plis pour la validation croisée

    Returns:
        list: Liste des caractéristiques sélectionnées à conserver
    """
    # Si une instance de Pipeline est passée comme modèle, extraire le modèle et le préprocesseur
    if isinstance(model, Pipeline):
        if preprocessor is None and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
        if 'model' in model.named_steps:
            model = model.named_steps['model']

    # Initialisation des caractéristiques
    if initial_features is None:
        features = list(X.columns)
    else:
        # Vérifier que les caractéristiques existent
        features = [f for f in initial_features if f in X.columns]

    # Vérifier qu'il y a suffisamment de caractéristiques
    if len(features) == 0:
        print("Aucune caractéristique valide fournie.")
        return []

    # Définir le nombre minimum de caractéristiques
    if min_features is None:
        min_features = max(3, int(0.1 * len(features)))

    # Créer le préprocesseur si non fourni
    if preprocessor is None:
        preprocessor = get_preprocessor(X)

    # Évaluation initiale
    initial_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    try:
        scores = cross_val_score(initial_model, X[features], y,
                                scoring='neg_root_mean_squared_error', cv=cv)
        best_rmse = -scores.mean()
        print(f"RMSE initiale: {best_rmse:.4f} avec {len(features)} caractéristiques")
    except Exception as e:
        print(f"Erreur lors de l'évaluation initiale: {e}")
        return features

    # Boucle d'élimination
    iteration = 1
    while len(features) > min_features:
        worst_rmse = float('inf')
        worst_feature = None
        current_rmse = []

        # Évaluer chaque caractéristique
        for f in features:
            temp_features = [v for v in features if v != f]

            # Éviter les erreurs si temp_features est vide
            if not temp_features:
                continue

            temp_model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            try:
                scores = cross_val_score(temp_model, X[temp_features], y,
                                        scoring='neg_root_mean_squared_error', cv=cv)
                rmse = -scores.mean()
                current_rmse.append(rmse)

                if rmse < worst_rmse:
                    worst_rmse = rmse
                    worst_feature = f
            except Exception as e:
                print(f"Erreur lors de l'évaluation sans {f}: {e}")
                continue

        # Vérifier s'il y a amélioration
        if not current_rmse:
            print("Aucune évaluation réussie, arrêt de la sélection")
            break

        relative_gain = (best_rmse - worst_rmse) / best_rmse
        print(f"Itération {iteration}: meilleure RMSE = {worst_rmse:.4f}, gain relatif = {relative_gain:.4f}, feature supprimée = {worst_feature}")

        # Décision de continuer ou non
        if relative_gain < rtol:
            print(f"Gain relatif ({relative_gain:.4f}) inférieur au seuil ({rtol}), arrêt de la sélection")
            break

        if worst_rmse < best_rmse:
            best_rmse = worst_rmse
            features.remove(worst_feature)
        else:
            print("Pas d'amélioration, arrêt de la sélection")
            break

        iteration += 1

    print(f"Sélection terminée: {len(features)} caractéristiques conservées avec RMSE = {best_rmse:.4f}")
    return features

def optimise_model(model, params, X, y, n_trials=100):
    """
    Optimise les hyperparamètres et sélectionne les features optimales pour le modèle.

    Args:
        model: Instance du modèle choisi (Lasso, ElasticNet, XGBoost, LightGBM)
        params (dict): Dictionnaire définissant l'espace des hyperparamètres à optimiser
        X (pd.DataFrame): DataFrame des prédicteurs
        y (pd.Series): Variable cible
        n_trials (int): Nombre d'essais pour l'optimisation Optuna

    Returns:
        tuple: (meilleurs_parametres, meilleures_features, etude_optuna)
    """
    # Validation croisée répétée pour plus de robustesse
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    # Première étape: réduction des features par importance
    print("Étape 1: Réduction initiale du nombre de features par importance...")
    # Identifier les colonnes à faible variance
    low_var_cols = identify_low_variance_features(X, display=False)
    print(f"Colonnes à faible variance identifiées: {len(low_var_cols)}")

    # Identifier les colonnes fortement corrélées
    corr_cols = identify_correlated_features(X, target=y.name if hasattr(y, 'name') else None)
    print(f"Colonnes fortement corrélées identifiées: {len(corr_cols)}")

    # Combiner les colonnes à supprimer
    cols_to_drop = list(set(low_var_cols + corr_cols))
    X_reduced = X.drop(columns=cols_to_drop, errors='ignore')
    print(f"DataFrame réduit de {X.shape[1]} à {X_reduced.shape[1]} colonnes")

    # Si trop de colonnes restantes, utiliser select_important_features
    if X_reduced.shape[1] > 50:
        print("Plus de 50 colonnes restantes, sélection des plus importantes...")
        important_cols = select_important_features(X_reduced, y)
        X_reduced = safe_feature_selection(X_reduced, important_cols)
        print(f"Nombre final de colonnes après sélection d'importance: {X_reduced.shape[1]}")

    # Fonction objective pour Optuna
    def objective(trial):
        # Définir les espaces de recherche pour chaque hyperparamètre
        param_grid = {}
        for name, conf in params.items():
            if conf['type'] == 'int':
                param_grid[name] = trial.suggest_int(name, *conf['range'])
            elif conf['type'] == 'float':
                param_grid[name] = trial.suggest_float(name, *conf['range'])
            elif conf['type'] == 'categorical':
                param_grid[name] = trial.suggest_categorical(name, conf['range'])
            elif conf['type'] == 'log':
                param_grid[name] = trial.suggest_float(name, *conf['range'], log=True)

        # Appliquer les paramètres au modèle
        model_trial = model.__class__(**{**model.get_params(), **param_grid})

        # Sélection fine des features par backward_stepwise_selection
        try:
            prepro = get_preprocessor(X_reduced)
            selected_features = backward_stepwise_selection(
                model_trial,
                X_reduced,
                y,
                preprocessor=prepro,
                cv=cv,
                min_features=max(3, int(0.05 * X_reduced.shape[1]))
            )

            # Sélection sécurisée des colonnes
            X_selected = safe_feature_selection(X_reduced, selected_features)

            # Si aucune colonne valide, utiliser toutes les colonnes réduites
            if X_selected.shape[1] == 0:
                X_selected = X_reduced
                selected_features = list(X_reduced.columns)

            # Enregistrer les features sélectionnées dans l'essai
            trial.set_user_attr('selected_features', selected_features)

            # Créer et évaluer le pipeline final
            final_prepro = get_preprocessor(X_selected)
            final_pipeline = Pipeline([
                ('preprocessor', final_prepro),
                ('model', model_trial)
            ])

            # Effectuer la validation croisée
            scores = cross_val_score(
                final_pipeline,
                X_selected,
                y,
                cv=cv,
                scoring='neg_root_mean_squared_error'
            )

            rmse = -np.mean(scores)
            return rmse

        except Exception as e:
            print(f"Erreur dans l'essai: {e}")
            # En cas d'erreur, retourner une valeur très élevée
            trial.set_user_attr('error', str(e))
            trial.set_user_attr('selected_features', list(X_reduced.columns))
            return float('inf')

    # Création et exécution de l'étude Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Récupération des meilleurs résultats
    best_params = study.best_params

    # Vérifier si les features sont disponibles dans le meilleur essai
    if 'selected_features' in study.best_trial.user_attrs:
        best_features = study.best_trial.user_attrs['selected_features']
    else:
        print("Aucune caractéristique n'a été enregistrée dans le meilleur essai. Utilisation")