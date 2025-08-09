def identify_correlated_features(df, target=None, threshold=0.8, matrix = False):
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
        upper = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=1).astype(bool))
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

        if matrix:
            # Afficher la matrice de corrélation
            plt.figure(figsize=(12, 8))
            sns.heatmap(upper, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title("Matrice de corrélation")
            plt.show()

        return list(to_drop) , df_numeric.drop(columns=list(to_drop), axis=1, errors='ignore').columns.tolist()

    except Exception as e:
        print(f"Erreur dans identify_correlated_features: {e}")
        return []

identify_correlated_features(train_data[keep] , '% Voix/Ins' , threshold=0.8 )