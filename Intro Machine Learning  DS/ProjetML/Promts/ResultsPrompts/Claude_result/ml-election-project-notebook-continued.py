# Recherche des meilleurs hyperparamètres avec validation croisée pour le Gradient Boosting
gb_cv = GridSearchCV(
    gb_model, 
    gb_params, 
    cv=5, 
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

print("\nEntraînement du modèle Gradient Boosting...")
gb_cv.fit(X_train, y_train_split)

print(f"Meilleurs paramètres pour Gradient Boosting: {gb_cv.best_params_}")
print(f"Meilleur score RMSE: {-gb_cv.best_score_:.4f}")

# Évaluation sur le set de validation
gb_pred = gb_cv.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
gb_r2 = r2_score(y_val, gb_pred)

print(f"Gradient Boosting - RMSE sur validation: {gb_rmse:.4f}")
print(f"Gradient Boosting - R² sur validation: {gb_r2:.4f}")

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_val, gb_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Gradient Boosting: Prédictions vs Valeurs réelles')
plt.savefig('gradientboosting_predictions.png')
plt.show()

# Analyse des résidus
residuals_gb = y_val - gb_pred
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals_gb, kde=True)
plt.title('Distribution des résidus - Gradient Boosting')

plt.subplot(1, 2, 2)
plt.scatter(gb_pred, residuals_gb)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus vs Prédictions - Gradient Boosting')
plt.tight_layout()
plt.savefig('gradientboosting_residuals.png')
plt.show()

# 4.3 Comparaison des modèles
# Créons un tableau comparatif
models = {
    'ElasticNet (Lasso-Ridge)': {
        'model': elastic_cv.best_estimator_,
        'rmse_cv': -elastic_cv.best_score_,
        'rmse_val': elastic_rmse,
        'r2_val': elastic_r2
    },
    'Gradient Boosting': {
        'model': gb_cv.best_estimator_,
        'rmse_cv': -gb_cv.best_score_,
        'rmse_val': gb_rmse,
        'r2_val': gb_r2
    }
}

comparison_df = pd.DataFrame({
    'RMSE (CV)': [models[m]['rmse_cv'] for m in models],
    'RMSE (Validation)': [models[m]['rmse_val'] for m in models],
    'R² (Validation)': [models[m]['r2_val'] for m in models]
}, index=models.keys())

print("\nComparaison des modèles:")
print(comparison_df)

# Visualisation graphique des performances
plt.figure(figsize=(10, 6))
comparison_df[['RMSE (CV)', 'RMSE (Validation)']].plot(kind='bar')
plt.title('Comparaison des performances (RMSE)')
plt.ylabel('RMSE (plus bas = meilleur)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison_rmse.png')
plt.show()

# 4.4 Analyse des caractéristiques importantes (pour le Gradient Boosting)
# Récupération des noms des features après transformation
feature_names = []
ohe = gb_cv.best_estimator_.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
numeric_cols = numeric_features
categorical_cols = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = numeric_cols + categorical_cols

# Extraction de l'importance des features
try:
    # Pour les modèles qui supportent feature_importances_
    importances = gb_cv.best_estimator_.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Limitation au top 20 features pour la lisibilité
    n_top_features = min(20, len(all_feature_names))
    
    plt.figure(figsize=(12, 8))
    plt.title('Importance des caractéristiques - Gradient Boosting')
    plt.bar(range(n_top_features), importances[indices][:n_top_features], align='center')
    plt.xticks(range(n_top_features), np.array(all_feature_names)[indices][:n_top_features], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("\nTop 10 caractéristiques les plus importantes:")
    for i in range(min(10, len(all_feature_names))):
        print(f"{all_feature_names[indices[i]]} : {importances[indices[i]]:.4f}")
except:
    print("Impossible d'extraire l'importance des caractéristiques pour ce modèle.")

# 5. OPTIMISATION AVANCÉE DU MODÈLE LE PLUS PERFORMANT

# Choisir le meilleur modèle pour l'optimisation avancée
best_model_name = comparison_df['RMSE (Validation)'].idxmin()
print(f"\nLe meilleur modèle est: {best_model_name}")
best_model = models[best_model_name]['model']

# 5.1 Ajustement fin des hyperparamètres
if best_model_name == 'Gradient Boosting':
    # Paramètres plus fins pour le Gradient Boosting
    fine_gb_params = {
        'regressor__n_estimators': [150, 200, 250, 300],
        'regressor__learning_rate': [0.03, 0.05, 0.07, 0.1],
        'regressor__max_depth': [3, 4, 5, 6],
        'regressor__min_samples_split': [4, 6, 8, 10],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }
    
    # Recherche des meilleurs hyperparamètres avec validation croisée
    fine_gb_cv = GridSearchCV(
        gb_model, 
        fine_gb_params, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    print("\nOptimisation fine du modèle Gradient Boosting...")
    fine_gb_cv.fit(X_train, y_train_split)
    
    print(f"Meilleurs paramètres après optimisation fine: {fine_gb_cv.best_params_}")
    print(f"Meilleur score RMSE après optimisation: {-fine_gb_cv.best_score_:.4f}")
    
    # Mettre à jour le meilleur modèle
    best_model = fine_gb_cv.best_estimator_
    
    # Évaluation sur le set de validation
    fine_gb_pred = fine_gb_cv.predict(X_val)
    fine_gb_rmse = np.sqrt(mean_squared_error(y_val, fine_gb_pred))
    fine_gb_r2 = r2_score(y_val, fine_gb_pred)
    
    print(f"Gradient Boosting optimisé - RMSE sur validation: {fine_gb_rmse:.4f}")
    print(f"Gradient Boosting optimisé - R² sur validation: {fine_gb_r2:.4f}")

elif best_model_name == 'ElasticNet (Lasso-Ridge)':
    # Paramètres plus fins pour l'ElasticNet
    fine_elastic_params = {
        'regressor__alpha': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'regressor__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    
    # Recherche des meilleurs hyperparamètres avec validation croisée
    fine_elastic_cv = GridSearchCV(
        elastic_model, 
        fine_elastic_params, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    print("\nOptimisation fine du modèle ElasticNet...")
    fine_elastic_cv.fit(X_train, y_train_split)
    
    print(f"Meilleurs paramètres après optimisation fine: {fine_elastic_cv.best_params_}")
    print(f"Meilleur score RMSE après optimisation: {-fine_elastic_cv.best_score_:.4f}")
    
    # Mettre à jour le meilleur modèle
    best_model = fine_elastic_cv.best_estimator_
    
    # Évaluation sur le set de validation
    fine_elastic_pred = fine_elastic_cv.predict(X_val)
    fine_elastic_rmse = np.sqrt(mean_squared_error(y_val, fine_elastic_pred))
    fine_elastic_r2 = r2_score(y_val, fine_elastic_pred)
    
    print(f"ElasticNet optimisé - RMSE sur validation: {fine_elastic_rmse:.4f}")
    print(f"ElasticNet optimisé - R² sur validation: {fine_elastic_r2:.4f}")

# 5.2 Vérification des limites des prédictions
# Les scores doivent être compris entre 0 et 100
def clip_predictions(predictions):
    """Contraindre les prédictions entre 0 et 100"""
    return np.clip(predictions, 0, 100)

# 6. ENTRAINEMENT SUR L'ENSEMBLE DES DONNÉES ET PRÉDICTIONS FINALES

# 6.1 Ré-entraînement du meilleur modèle sur toutes les données d'entraînement
print("\nEntraînement du modèle final sur toutes les données...")
best_model.fit(train_features, y_train)

# 6.2 Prédictions sur l'ensemble de test
final_predictions = best_model.predict(test_features)

# Contraindre les prédictions entre 0 et 100
final_predictions = clip_predictions(final_predictions)

# Vérifier la distribution des prédictions
plt.figure(figsize=(10, 6))
sns.histplot(final_predictions, kde=True)
plt.title('Distribution des prédictions finales')
plt.axvline(final_predictions.mean(), color='r', linestyle='--', 
           label=f'Moyenne: {final_predictions.mean():.2f}%')
plt.xlabel('% Voix/Ins prédit')
plt.ylabel('Fréquence')
plt.legend()
plt.savefig('predictions_distribution.png')
plt.show()

# 6.3 Création du fichier de soumission Kaggle
submission = pd.DataFrame({
    'CodeINSEE': test_features['CodeINSEE'],
    'Prediction': final_predictions
})

# Sauvegarde du fichier de soumission
submission.to_csv('submission_file.csv', index=False)

print(f"\nFichier de soumission créé avec {len(submission)} prédictions.")
print("Statistiques des prédictions:")
print(submission['Prediction'].describe())

# 7. ANALYSE SPATIALE DES PRÉDICTIONS

# 7.1 Créer une carte des prédictions par département
# Fusion des prédictions avec les données géographiques
pred_geo = pd.merge(
    submission,
    communes_france[['code_insee', 'dep_code', 'dep_nom', 'latitude_centre', 'longitude_centre']],
    left_on='CodeINSEE',
    right_on='code_insee',
    how='left'
)

# Calcul des moyennes par département
dept_avg = pred_geo.groupby('dep_code')['Prediction'].mean().reset_index()
dept_avg = dept_avg.rename(columns={'Prediction': 'Score_Moyen_Predit'})

print("\nScore moyen prédit par département (top 10):")
print(dept_avg.sort_values('Score_Moyen_Predit', ascending=False).head(10))

print("\nScore moyen prédit par département (bottom 10):")
print(dept_avg.sort_values('Score_Moyen_Predit').head(10))

# Visualisation de la distribution des scores par département
plt.figure(figsize=(14, 8))
sns.boxplot(x='dep_code', y='Prediction', data=pred_geo.sort_values('Score_Moyen_Predit', ascending=False).head(20))
plt.title('Distribution des scores prédits par département (Top 20 départements)')
plt.xticks(rotation=90)
plt.xlabel('Code Département')
plt.ylabel('Score prédit (%)')
plt.tight_layout()
plt.savefig('dept_predictions.png')
plt.show()

# 8. SAUVEGARDER LE MODÈLE POUR UTILISATION FUTURE

# Sauvegarde du modèle final
joblib.dump(best_model, 'best_model.pkl')
print("\nModèle sauvegardé sous 'best_model.pkl'")

# Sauvegarder également les noms des features pour référence future
with open('feature_names.txt', 'w') as f:
    for feature in train_features.columns:
        f.write(f"{feature}\n")

print("Noms des features sauvegardés sous 'feature_names.txt'")
print("\nProjet terminé!")
