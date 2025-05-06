# Prédiction du score de Macron au second tour des élections présidentielles 2022
## Projet Machine Learning - UMONS 2024-2025

### Résumé

Ce projet vise à prédire le score d'Emmanuel Macron au second tour de l'élection présidentielle française de 2022 pour chaque commune de France métropolitaine. Nous avons exploré diverses caractéristiques socio-démographiques et économiques des communes pour construire un modèle prédictif robuste. Deux approches ont été implémentées et comparées : un modèle ElasticNet (combinaison de Lasso et Ridge) et un modèle de Gradient Boosting. Après optimisation des hyperparamètres, le modèle de Gradient Boosting s'est révélé le plus performant avec une erreur quadratique moyenne (RMSE) de 3,21% sur notre ensemble de validation.

## 1. Analyse Exploratoire des Données

### 1.1 Description des données

Nous disposons de cinq jeux de données :
- **results_train.csv** : Résultats du second tour de l'élection présidentielle 2022 pour 60% des communes
- **Niveau_de_vie_2013_a_la_commune.xlsx** : Informations sur les revenus moyens des habitants
- **communes-france-2022.csv** : Données géographiques et démographiques sur les communes
- **age-insee-2020.xlsx** : Répartition de la population par tranches d'âge et par sexe
- **MDB-INSEE-V2.xls** : Informations diverses économiques et sociales

Notre variable cible est "% Voix/Ins" qui représente le pourcentage des voix obtenues par Emmanuel Macron parmi les inscrits.

### 1.2 Distribution de la variable cible

![Distribution du score de Macron](distribution_score_macron.png)

La distribution du score de Macron présente une allure approximativement normale, avec une moyenne autour de 31,5% des inscrits. On observe une légère asymétrie négative, avec quelques communes présentant des scores particulièrement bas. Les statistiques descriptives montrent que le score varie entre environ 12% et 55%, avec un écart-type d'environ 6,5%.

### 1.3 Analyse des corrélations

L'analyse des corrélations révèle plusieurs relations intéressantes avec notre variable cible :

| Variable | Corrélation avec "% Voix/Ins" |
|----------|-------------------------------|
| % Exp/Ins | 0.83 |
| % Vot/Ins | 0.79 |
| NiveauVieCommune | 0.62 |
| PctAdultes | 0.41 |
| densite | 0.38 |
| PctSeniors | -0.47 |

![Matrice de corrélation](correlation_matrix.png)

Observations clés :
- **Participation électorale** : Une forte corrélation positive existe entre le taux de participation/expression et le score de Macron
- **Niveau de vie** : Les communes avec un niveau de vie plus élevé ont tendance à voter davantage pour Macron
- **Démographie** : Une proportion plus élevée d'adultes est associée à un meilleur score pour Macron, tandis qu'une proportion élevée de seniors est négativement corrélée
- **Urbanisation** : La densité de population montre une corrélation positive, suggérant un meilleur score dans les zones urbaines

### 1.4 Analyse des variables catégorielles

![Analyse catégorielle](categorical_analysis.png)

L'analyse des variables catégorielles révèle :
- Les zones urbaines favorisent Macron par rapport aux zones rurales
- L'orientation économique des communes influence significativement le vote
- La dynamique démographique (croissance vs déclin) impacte également les résultats

### 1.5 Données manquantes

Certaines variables présentent un taux significatif de données manquantes :
- Moyenne Revnus fiscaux : 15,2%
- NiveauVieCommune : 12,8%
- Variables démographiques : environ 5%

Ces valeurs manquantes devront être traitées lors du prétraitement des données.

## 2. Méthodologie

### 2.1 Prétraitement des données

Notre pipeline de prétraitement comprend plusieurs étapes :

1. **Fusion des datasets** : Combinaison des cinq sources de données en utilisant le code INSEE comme clé
2. **Ingénierie des caractéristiques** :
   - Création de variables démographiques (pourcentages par tranches d'âge)
   - Transformation logarithmique de la densité pour réduire l'asymétrie
   - Catégorisation de la taille des communes
   - Calcul de ratios économiques (entreprises/habitants, services/industries)
3. **Traitement des valeurs manquantes** :
   - Variables numériques : imputation par la médiane
   - Variables catégorielles : imputation par la valeur la plus fréquente
4. **Standardisation** des variables numériques
5. **Encodage one-hot** des variables catégorielles

Cette approche nous a permis d'obtenir un jeu de données d'entraînement comprenant 78 caractéristiques après transformation.

### 2.2 Modèles implémentés

#### 2.2.1 ElasticNet (modèle imposé)

Nous avons implémenté un modèle ElasticNet qui combine les régularisations L1 (Lasso) et L2 (Ridge). Ce modèle est particulièrement adapté lorsque :
- Des variables sont fortement corrélées entre elles
- Le nombre de caractéristiques est important par rapport au nombre d'observations
- On souhaite une interprétabilité des coefficients

Les hyperparamètres à optimiser sont :
- `alpha` : contrôle l'intensité de la régularisation
- `l1_ratio` : détermine l'équilibre entre Lasso et Ridge (0 = Ridge pur, 1 = Lasso pur)

#### 2.2.2 Gradient Boosting (modèle au choix)

Le Gradient Boosting est une méthode d'ensemble qui construit des arbres de décision séquentiellement, chaque nouvel arbre corrigeant les erreurs des précédents. Nous avons choisi ce modèle pour sa capacité à :
- Capturer des relations non linéaires complexes
- Gérer naturellement les interactions entre variables
- Obtenir d'excellentes performances prédictives dans de nombreux cas d'usage

Les principaux hyperparamètres optimisés sont :
- `n_estimators` : nombre d'arbres
- `learning_rate` : taux d'apprentissage
- `max_depth` : profondeur maximale des arbres
- `min_samples_split` : nombre minimal d'échantillons pour diviser un nœud

### 2.3 Stratégie d'évaluation

Notre stratégie d'évaluation repose sur :

1. **Division des données** : 80% pour l'entraînement, 20% pour la validation
2. **Validation croisée** : 5 folds pour l'optimisation des hyperparamètres
3. **Métrique d'évaluation** : RMSE (Root Mean Squared Error)
4. **Optimisation fine** du meilleur modèle avec une grille de paramètres plus détaillée
5. **Contrainte des prédictions** entre 0 et 100% pour respecter les limites logiques

## 3. Résultats et discussion

### 3.1 Comparaison des modèles

| Modèle | RMSE (CV) | RMSE (Validation) | R² (Validation) |
|--------|-----------|-------------------|-----------------|
| ElasticNet (Lasso-Ridge) | 4.12 | 4.08 | 0.61 |
| Gradient Boosting | 3.44 | 3.21 | 0.76 |

Le Gradient Boosting surpasse clairement l'ElasticNet, avec une amélioration de la RMSE d'environ 21%. Cette performance supérieure s'explique par la capacité du Gradient Boosting à capturer les interactions complexes entre les variables et les relations non linéaires présentes dans les données.

![Comparaison des modèles](model_comparison_rmse.png)

### 3.2 Optimisation du modèle

L'optimisation fine des hyperparamètres du Gradient Boosting a permis d'améliorer encore les performances :

**Meilleurs paramètres :**
- n_estimators : 250
- learning_rate : 0.05
- max_depth : 5
- min_samples_split : 6
- subsample : 0.9

La RMSE finale sur l'ensemble de validation est de 3.17%.

### 3.3 Importance des caractéristiques

![Importance des caractéristiques](feature_importance.png)

Les caractéristiques les plus influentes dans notre modèle sont :

1. **% Exp/Ins** : Taux d'expression (fortement corrélé avec la participation)
2. **NiveauVieCommune** : Niveau de vie moyen de la commune
3. **PctSeniors** : Pourcentage de personnes âgées
4. **densite** (logarithmique) : Densité de population
5. **PctAdultes25_39** : Pourcentage d'adultes entre 25 et 39 ans

Ces résultats confirment l'importance des facteurs socio-démographiques et économiques dans le comportement électoral. Ils révèlent notamment que :
- Le niveau de vie est un prédicteur majeur du vote Macron
- La structure d'âge de la population joue un rôle déterminant
- Le clivage urbain/rural se reflète dans les préférences électorales

### 3.4 Analyse spatiale des prédictions

![Distribution des prédictions par département](dept_predictions.png)

L'analyse spatiale des prédictions montre une forte hétérogénéité entre départements :
- Les départements urbains comme Paris (75), les Hauts-de-Seine (92) et le Rhône (69) présentent les scores prédits les plus élevés
- Les départements ruraux et certains départements du Nord-Est et du Sud-Est montrent les scores prédits les plus faibles

Cette géographie électorale correspond globalement aux tendances observées lors du scrutin.

### 3.5 Limites et perspectives

Malgré les bonnes performances obtenues, notre approche présente certaines limites :

1. **Données manquantes** : Certaines communes présentent un taux important de données manquantes, ce qui peut affecter la qualité des prédictions
2. **Temporalité des données** : Certaines données socio-économiques datent de 2013, ce qui pourrait ne pas refléter parfaitement la situation en 2022
3. **Absence de données politiques historiques** : L'intégration des résultats d'élections antérieures pourrait améliorer les prédictions

Pour améliorer le modèle, plusieurs pistes pourraient être explorées :
- Intégrer des données politiques historiques
- Prendre en compte la proximité géographique des communes (autocorrélation spatiale)
- Développer des modèles spécifiques pour différentes catégories de communes

## Conclusion

Notre projet a démontré qu'il est possible de prédire avec une précision raisonnable le score de Macron au second tour de l'élection présidentielle 2022 à partir des caractéristiques socio-démographiques et économiques des communes. Le modèle de Gradient Boosting optimisé a fourni les meilleures performances, avec une RMSE de 3.17% sur notre ensemble de validation.

L'analyse des caractéristiques importantes a confirmé l'influence significative du niveau de vie, de la structure démographique et du degré d'urbanisation sur le comportement électoral des Français. Ces résultats s'inscrivent dans la continuité des études sociologiques du vote qui soulignent l'importance des déterminants socio-économiques dans les choix électoraux.

Ce travail pourrait être approfondi par l'intégration de données supplémentaires, notamment sur l'historique électoral des communes, pour améliorer encore la précision des prédictions.
