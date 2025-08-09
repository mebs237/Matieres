
### définition des hyperparmètres à optimiser
Enet_space = {
       'alpha': {'type': 'float', 'range': [0.01, 0.5]},
       'l1_ratio': {'type': 'float', 'range': [0.2, 0.8]},
       'max_iter': {'type': 'int', 'range': [1000, 2000]},
       'fit_intercept': {'type': 'categorical', 'range': [True]}
   }

### Recherche des meilleurs hyperparamètres
Enet_best_params, Enet_selected_features, Enet_study = optimise_model(Enet_model,Enet_space, X_train, y_train,model_name="ElasticNet")

### Finalisation du modèle et évaluation
X_train_enet = X_train[Enet_selected_features]
X_test_enet = X_test[Enet_selected_features]

# le modèle avec les hyperparamètres optimaux
Enet_model.set_params(**Enet_best_params)
enet_pipeline = Pipeline([
    ('preprocessor', get_preprocessor(X_train_enet)),
    ('model', Enet_model)
])
# fit sur le train set
enet_pipeline.fit(X_train_enet, y_train)

# prédiction sur l'ensemble de validation
y_pred_enet = enet_pipeline.predict(X_test_enet)










### définition des hyperparmètres à optimiser
xgb_space = {
    'n_estimators': {'type': 'int', 'range': [100, 300]},
    'max_depth': {'type': 'int', 'range': [3, 6]},
    'learning_rate': {'type': 'float', 'range': [0.02, 0.1]},
    'subsample': {'type': 'float', 'range': [0.7, 1.0]},
    'colsample_bytree': {'type': 'float', 'range': [0.7, 1.0]},
    'reg_alpha': {'type': 'float', 'range': [0, 1]},
    'reg_lambda': {'type': 'float', 'range': [0, 1]}
    }

### Recherche des meilleurs hyperparamètres
xgb_best_params, xgb_selected_features, xgb_study = optimise_model(xgb_model, xgb_space, X_train, y_train,model_name="XGBoost")

### Finalisation du modèle et évaluation
X_train_xgb = X_train[xgb_selected_features]
X_test_xgb = X_test[xgb_selected_features]

# le modèle avec les hyperparamètres optimaux
xgb_model.set_params(**xgb_best_params)
xgb_pipeline = Pipeline([
    ('preprocessor', get_preprocessor(X_train_xgb)),
    ('model', xgb_model)
])

# fit sur l'ensemble d'entrainement
xgb_pipeline.fit(X_train_xgb, y_train)
# prediction sur l'ensemble de validation
y_pred_xgb = xgb_pipeline.predict(X_test_xgb)






### initialisation du modèle
lasso_model = Lasso(random_state=RANDOM_STATE)

### définition des hyperparmètres à optimiser
lasso_space = {
       'alpha': {'type': 'float', 'range': [0.05 , 0.7]},
       'max_iter': {'type': 'int', 'range': [1000, 2000]},
       'fit_intercept': {'type': 'categorical', 'range': [True]}
   }



### Recherche des meilleurs hyperparamètres
lasso_best_params, lasso_selected_features, lasso_study = optimise_model(lasso_model,lasso_space, X_train, y_train,model_name="Lasso")

### Finalisation du modèle et évaluation
X_train_lasso = X_train[lasso_selected_features]
X_test_lasso = X_test[lasso_selected_features]

# le modèle avec les hyperparamètres optimaux
lasso_model.set_params(**lasso_best_params)
lasso_pipeline = Pipeline([
    ('preprocessor', get_preprocessor(X_train_lasso)),
    ('model', lasso_model)
])

# fit sur l'ensemble d'entrainement
lasso_pipeline.fit(X_train_lasso, y_train)
# prediction sur l'ensemble de validation
y_pred_lasso = lasso_pipeline.predict(X_test_lasso)