def get_feature_names(preprocessor, X):
    """
    Retourne les noms des colonnes après le prétraitement
    """

    X=X.copy()
    cat_col = X.select_dtypes(include=['object','category','bool']).columns
    num_col = X.select_dtypes(include=['int64', 'float64',]).columns


    for col in cat_col:
      if X[col].apply(type).nunique() > 1:
        X[col] = X[col].astype(str)

    preprocessor.fit(X)

    cat_col = X.select_dtypes(include=['object','category','bool']).columns
    num_col = X.select_dtypes(include=['int64', 'float64',]).columns



    feature_names = list(num_col)


    # Add categorical feature names if there are any categorical columns
    if len(cat_col) > 0:
        cat_transformer = preprocessor.transformers[1][1]
        onehot = cat_transformer.named_steps['onehot']
        cat_features = onehot.get_feature_names_out(cat_col)
        feature_names.extend(cat_features)

    return feature_names

print(get_feature_names(get_preprocessor(X[keep]), X[keep]))