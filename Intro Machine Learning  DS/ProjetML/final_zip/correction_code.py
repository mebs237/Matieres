---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[20], line 1
----> 1 niveau_matrix = mutual_info_matrix(niveau_vie_df)

File c:\Users\lance\OneDrive\Documents\Cours\Intro Machine Learning  DS\ProjetML\final_zip\utilitaires.py:493, in mutual_info_matrix(df, target)
    491     mi = np.nan
    492 else:
--> 493     mi = mutual_info_norm(sub_df[c1], sub_df[c2])
    494 matrix.loc[c1, c2] = mi
    495 matrix.loc[c2, c1] = mi

File c:\Users\lance\OneDrive\Documents\Cours\Intro Machine Learning  DS\ProjetML\final_zip\utilitaires.py:441, in mutual_info_norm(x, y, n_bins, strategy)
    439     x = discretize_column(x, n_bins=n_bins, strategy=strategy)
    440 if pd.api.types.is_numeric_dtype(y):
--> 441     y = discretize_column(y, n_bins=n_bins, strategy=strategy)
    443 try :
    444     # calcul de la mutual_information
    445     mi = mutual_info_score(x,y)

File c:\Users\lance\OneDrive\Documents\Cours\Intro Machine Learning  DS\ProjetML\final_zip\utilitaires.py:128, in discretize_column(col, n_bins, strategy)
    126 discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    127 col = discretizer.fit_transform(col.values.reshape(-1, 1)).flatten()
--> 128 return pd.Series(col, index=col.index)

AttributeError: 'numpy.ndarray' object has no attribute 'index'
# code corrigé
def discretize_column(col, n_bins=10, strategy='quantile'):
    from sklearn.preprocessing import KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    values = discretizer.fit_transform(col.values.reshape(-1, 1)).flatten()
    return pd.Series(values, index=col.index)  # ✅ Corrigé ici
