Méthode d'exploration des données pour chaque dataframe
 ==================================================

# 1. Identification et élimination des colonnes à faible information

*  **Objectif** :
    Supprimer les colonnes qui ne fournissent pratiquement aucune information utile pour la prédiction.
* **Méthodes** :
    utiliser la fonction `` low_information_cols`` , elle dectecte :
        - les colonnes avec un ratio élévé de valeurs manquantes
        - les colonnes quasi_constantes (celles qui on une variance très faible)
        - les colonnes à faibles entropie
    Elle utilise les fonctions  ``to_missing_values`` , ``quasi_constant_cols`` et ``low_entropy_cols`` pour identifier ces colonnes.
* **Visualisation** :
    Afficher des histogrammes et des box plots pour les colonnes numériques identifiées.
    Afficher des bar plots pour les colonnes catégorielles identifiées.
* **Décision** :
    Confirmer visuellement que les colonnes identifiées sont effectivement peu informatives.
    Supprimer ces colonnes du DataFrame.
# 2. Regroupement des colonnes avec une logique métier

* **Objectif** : Créer de nouvelles colonnes plus significatives en combinant des colonnes existantes sur la base de votre connaissance du domaine.
* **Méthodes** :
    Identifier les groupes de colonnes qui peuvent être combinés pour créer de nouvelles caractéristiques (par exemple, calcul de ratios, sommes, différences, etc.).
    Créer de nouvelles colonnes en effectuant les opérations appropriées sur les colonnes existantes.

# 3. Réduction des colonnes "inutiles" avec l'ACP

* **Objectif** : Réduire la dimensionnalité des données en remplaçant un groupe de colonnes par un ensemble réduit de composantes principales.
* **Méthodes** :
    Identifier les colonnes que vous jugez "inutiles" (c'est-à-dire, peu susceptibles d'être prédictives ou informatives).
    Appliquer l'ACP sur ces colonnes via ``apply_pca`` pour créer un ensemble de composantes principales.
    Sélectionner un nombre de composantes principales qui capture une proportion significative de la variance totale.
    Remplacer les colonnes originales par les composantes principales sélectionnées.
# 4. Élimination des paires de colonnes fortement corrélées

* **Objectif** : Réduire la redondance dans les données en supprimant les colonnes qui sont fortement dépendantes entre elles.
* **Méthodes** :
    Utiliser la fonction ``get_dependant_pairs`` pour identifier les paires de colonnes fortement dépendantes.
    Utiliser la fonction ``remove_redundant_cols`` qui pour chaque paire de colonnes fortement dépendantes , supprime la colonne qui est la moins dépendante avec la variable cible (ou, si aucune variable cible n'est disponible, supprimer la 1ère colonne de la paire).

# 5. Sélection finale des colonnes à conserver

* **Objectif** : Obtenir une liste finale des colonnes à conserver pour la modélisation.
* **Méthodes** :
    La liste des colonnes restantes après les étapes précédentes est votre liste finale de colonnes à conserver.
    Points importants :

* **Ordre des étapes** : L'ordre des étapes est important. Il est préférable d'éliminer d'abord les colonnes à faible information, puis de regrouper les colonnes avec une logique métier, puis de réduire les colonnes "inutiles" avec l'ACP, et enfin d'éliminer les paires de colonnes fortement corrélées.