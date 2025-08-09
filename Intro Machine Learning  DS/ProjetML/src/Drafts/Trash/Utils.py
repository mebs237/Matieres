import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , RobustScaler , OneHotEncoder
from sklearn.impute import  SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV , LassoCV
from xgboost import XGBRegressor
import optuna

# données  aléatoires
df = pd.DataFrame(np.ones(shape=(500,50)))
#  Split des données
y = df['col_cible']
X = df.drop(columns=['col_cible'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def preselect_feature(data):
    selector = VarianceThreshold(threshold=0.01)
    x_var = selector.fit_transform(data)
    preselect_col = data.columns[selector.get_support()]
    return preselect_col
# préprocesseur

def prepro(xtrain):
    """
        définit le préprocesseur pour le prétraitement des données
    """
    num_col = xtrain.select_dtypes(include=['int64', 'float64']).columns
    cat_col = xtrain.select_dtypes(include=['object']).columns

    # transformer des variables numériques
    num_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',RobustScaler())
    ])
    # transformer des variables catégorielles
    cat_transfomer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])
    # transformer des colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',num_transformer , num_col),
            ('cat',cat_transfomer,cat_col)
        ]
    )

    return preprocessor
#  Fonction générique de sélection backward stepwise

def backward_stepwise_selection(model, X, y, min_features=5, cv=3, tol=0.01):
    """
    Version améliorée avec critère d'arrêt dynamique :
    - Arrêt si suppression d'une feature ne dégrade pas les performances de plus que 'tol'
    - Conservation d'un nombre minimum de features ('min_features')
    """
    features = list(X.columns)
    best_rmse = float('inf')

    while len(features) > min_features:
        worst_rmse = float('inf')
        worst_feature = None
        current_rmse = []

        # Évaluation de chaque feature
        for f in features:
            temp_features = [v for v in features if v != f]
            scores = cross_val_score(model, X[temp_features], y,
                                    scoring='neg_root_mean_squared_error', cv=cv)
            rmse = -scores.mean()
            current_rmse.append(rmse)

            if rmse < worst_rmse:
                worst_rmse = rmse
                worst_feature = f

        # Critère d'arrêt dynamique
        if (worst_rmse - best_rmse) < tol and len(features) <= min_features:
            break

        if worst_rmse < best_rmse:
            best_rmse = worst_rmse
            features.remove(worst_feature)
        else:
            break

    return features

#  Fonctions d'optimisation séparées pour chaque modèle

def optimize_model(model,params,xtrain,ytrain):
    """
        fonction pour l'optimisation d'un modèle avec les meilleurs features
    """

    def objective(trial):
        param = { k : trial.suggest_type(k,v) for k,v in params.items}
        pip = Pipeline(steps=[
            ('preprocessor',prepro(xtrain)),
            ('regressor',model(**param))
        ])
        selected_features = backward_stepwise_selection(model,
                                                        xtrain,
                                                        ytrain)
        cv = RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)
        try :
            scores = cross_val_score(pip,
                                    xtrain[selected_features],
                                    ytrain,
                                    scoring='neg_root_mean_squared_error',
                                    cv=cv,
                                    error_score='raise')
            # verifier s'il y a des NaN
            if np.isnan(np.mean(scores)):
                return float('inf')
        except Exception as e:
            print(f"Error in trial:{str(e)}")
            return float('inf')

        trial.set_user_attr("features",selected_features)
        return -np.mean(scores)
    def log_fail_trail(study , trial):
        if trial.state == optuna.trail.TrialState.Fail:
            print(f"Trial {trial.number} failed due to {trial.value}")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective,
                   n_trials=100,
                   show_progress_bar=True,
                   callbacks = [log_fail_trail])
    best_features = study.best_trial.user_attrs["features"]
    best_params = study.best_params

    return {'best_param':best_params,
            'best_features': best_features,
            'study':study}

def enet_objective(trial):
    params = {
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        "alphas": trial.suggest_float("alpha", 1e-4, 1e2, log=True)
    }
    model = ElasticNetCV(**params, cv=5, random_state=42)

    # Sélection de features spécifique à ElasticNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    selected_features = backward_stepwise_selection(model, pd.DataFrame(X_scaled), y_train)

    trial.set_user_attr("features", selected_features)
    return -np.mean(cross_val_score(model, X_scaled[:, selected_features], y_train,
                                   scoring='neg_root_mean_squared_error', cv=KFold(5)))

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e2, log=True),
    }
    model = Pipeline(steps=[
        ('preprocessor',prepro(X)),
        ('regressor',XGBRegressor(**params, random_state=42))
    ])

    # Sélection de features spécifique à XGBoost
    selected_features = backward_stepwise_selection(model, X_train, y_train)
    cv = RepeatedKFold(n_splits=5,n_repeats=2,random_state=42)
    # Validation finale
    scores = cross_val_score(model,
                             X_train[selected_features],
                             y_train,
                             scoring='neg_root_mean_squared_error', cv=cv,
                             error_score='raise')

    trial.set_user_attr("features", selected_features)
    return -np.mean(scores)

#  Optimisation séparée
xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=30)

enet_study = optuna.create_study(direction="minimize")
enet_study.optimize(enet_objective, n_trials=30)

# Entraînement des modèles finaux avec leurs propres features

## XGBoost
xgb_features = xgb_study.best_trial.user_attrs["features"]
xgb_model = XGBRegressor(**xgb_study.best_params)
xgb_model.fit(X_train[xgb_features], y_train)

## ElasticNet
enet_features = enet_study.best_trial.user_attrs["features"]
scaler = StandardScaler().fit(X_train)
enet_model = ElasticNetCV(cv=5).fit(scaler.transform(X_train)[:, enet_features], y_train)

# 6. Comparaison équitable
def evaluate(model, features, scaler=None):
    if scaler:
        X_test_ = scaler.transform(X_test)
        return model.predict(X_test_[:, features])
    else:
        return model.predict(X_test[features])

results = {
    "XGBoost": report(y_test, evaluate(xgb_model, xgb_features)),
    "ElasticNet": report(y_test, evaluate(enet_model, enet_features, scaler))
}

print(pd.DataFrame(results))