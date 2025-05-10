def combined_feature_selection(X_data, y, n_features=20):
    X = X_data.copy()
    drop, keep = remove_quasi_constant_features(X)
    X = X[keep]
    X = X.replace([np.inf, -np.inf], np.nan)

    num_col = X.select_dtypes(include=[np.number, 'float64','int64']).columns
    cat_col = X.select_dtypes(include=['object', 'category', 'bool']).columns

    X[cat_col] = X[cat_col].astype(str)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = make_column_transformer(
        (num_pipeline, num_col),
        (cat_pipeline, cat_col),
        remainder='drop',
        verbose_feature_names_out=False,
        n_jobs=-1
    )

    # Apply preprocessing
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out(input_features=keep)

    # Define models
    xgb = XGBRegressor(tree_method='hist',
                       n_estimators=50,
                       max_depth=6,
                       max_features=n_features,
                       random_state=RANDOM_STATE)
    lasso = LassoCV(cv=5)
    rf = RandomForestRegressor(n_estimators=50,
                               max_depth=6,
                               max_features=n_features,
                               random_state=RANDOM_STATE)

    # Fit models
    xgb.fit(X_transformed, y)
    lasso.fit(X_transformed, y)
    rf.fit(X_transformed, y)

    # Get importances or coefficients
    xgb_importance = xgb.feature_importances_
    lasso_coef = np.abs(lasso.coef_)
    rf_importance = rf.feature_importances_

    # Normalize
    def normalize(arr):
        return arr / np.max(arr) if np.max(arr) != 0 else arr

    # scores nomalisés
    scores_norm = [
        normalize(xgb_importance),
        normalize(lasso_coef) ,
        normalize(rf_importance)
    ]
    scores = np.vstack(scores_norm)
    scores = np.median(scores, axis=0)

    # Select top features
    top_indices = np.argsort(scores)[::-1][:n_features]
    selected_features = [feature_names[i] for i in top_indices]

    return selected_features

