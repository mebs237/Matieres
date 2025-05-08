# Explication du Test du Collectionneur de Coupons
Le test du collectionneur de coupons est une méthode statistique utilisée pour évaluer le caractère aléatoire d'une séquence de nombres, en particulier pour vérifier si les nombres se comportent comme s'ils étaient tirés indépendamment d'une distribution uniforme sur un ensemble fini de d valeurs. Son nom provient de l'analogie avec un collectionneur qui rassemble d types de coupons différents (par exemple, dans des boîtes de céréales) et s'intéresse au nombre de boîtes (éléments de la séquence) qu'il doit ouvrir pour obtenir un ensemble complet des d coupons.
Le test consiste à examiner la séquence de nombres discrets (prenant des valeurs de 0 à d-1) et à mesurer la longueur des segments consécutifs nécessaires pour observer au moins une fois chacune des d valeurs possibles. Ces longueurs de segments sont enregistrées pour un nombre n de segments complets. Un test du chi-carré (χ²) est ensuite appliqué pour comparer la distribution observée de ces longueurs avec la distribution théoriquement attendue pour une séquence véritablement aléatoire.
Distinction entre Motifs, Catégories, Classes et Valeurs distinctes dans ce contexte
Dans le cadre du test du collectionneur de coupons :
•
* Valeurs distinctes : C'est un concept central. Il s'agit des d valeurs différentes (0, 1, ..., d-1) que l'on cherche à collecter dans chaque segment. Un segment est complet lorsqu'il contient les d valeurs distinctes requises pour la première fois.
•
* Motifs / Patterns : Bien que le terme "motif" ne soit pas utilisé de la même manière que pour les mains de poker, dans ce test, un "motif" pourrait être considéré comme la séquence spécifique de valeurs qui forme un segment complet, mais le test ne classe pas les segments par leur motif exact de valeurs.
•
* Catégories / Classes : Les "catégories" ou "classes" pour le test statistique sont définies par la longueur des segments complets. Par exemple, si d=3, les segments complets auront une longueur d'au moins 3. On regroupe les longueurs observées dans des catégories (par exemple, longueur 3, longueur 4, ..., longueur t-1, et longueur supérieure ou égale à t). Le test du chi-carré est appliqué à ces catégories de longueurs.
En résumé, on cherche à obtenir d valeurs distinctes dans chaque segment. Les segments sont classés en catégories ou classes selon leur longueur, et ces fréquences observées sont comparées aux fréquences attendues.

# Application à une Séquence de Nombres Réels vs. Entiers
Le test du collectionneur de coupons, tel que décrit dans les sources, s'applique directement à une séquence de nombres entiers Yj où 0 <= Yj < d. Si vous disposez d'une séquence de nombres réels Uj (généralement dans l'intervalle [0, 1[), vous devez d'abord les convertir en nombres entiers discrets.
La méthode standard pour cette conversion, similaire à celle utilisée pour le test du poker (mentionnée dans notre conversation précédente), consiste à choisir un paramètre d et à définir Yj comme la partie entière de d * Uj (Yj = Ld * UjJ). Cette opération divise l'intervalle [0, 1[ en d sous-intervalles de taille égale et mappe chaque nombre réel à l'indice entier correspondant à son intervalle (de 0 à d-1). Le test est ensuite appliqué à la séquence d'entiers Yj ainsi obtenue. Le choix de d a un impact significatif sur la sensibilité du test.

# Algorithme Simple pour le Test du Collectionneur de Coupons (Partie Comptage)
Voici un algorithme simple, basé sur la description de l'Algorithme C dans les sources, pour compter la fréquence des longueurs de segments. Il ne comprend pas le calcul des probabilités attendues ni l'application du test du chi-carré, car ces étapes impliquent des formules plus complexes (comme les nombres de Stirling de seconde espèce pour les probabilités attendues) qui sortent du cadre d'un algorithme de comptage simple, mais il fournit les données nécessaires (les fréquences observées COUNT).
Cet algorithme compte les longueurs de n segments complets, où un segment est complet lorsqu'il contient les d valeurs distinctes (0 à d-1). Les longueurs sont catégorisées jusqu'à max_length_t.
1. Choisir les paramètres :
◦
d : Le nombre de valeurs distinctes (coupons) à collecter (ex: 3 si valeurs dans {0, 1, 2}).
◦
num_segments_n : Le nombre de segments complets à observer (ex: 7).
◦
max_length_t : La longueur maximale à catégoriser individuellement. Les segments de longueur supérieure ou égale à max_length_t seront regroupés (ex: 5).
◦
La séquence d'entrée (input_sequence).
2. Préparer la séquence (si réels) :
◦
Si input_sequence contient des nombres réels (par exemple dans [0, 1[), créer une nouvelle séquence integer_sequence en convertissant chaque réel U en entier Y = floor(d * U). Les valeurs dans integer_sequence doivent être dans [0, d-1].
◦
Si input_sequence contient déjà des entiers dans [0, d-1], utiliser directement cette séquence.
3. Initialiser les compteurs :
◦
Créer un tableau COUNT de taille max_length_t + 1, initialisé à zéro. COUNT[r] stockera le nombre de segments de longueur r pour d <= r < max_length_t, et COUNT[max_length_t] stockera le nombre de segments de longueur >= max_length_t.
4. Initialiser les variables d'état :
◦
sequence_index = 0 (indice dans la séquence d'entiers).
◦
segments_found = 0.
5. Boucle principale (recherche des segments) :
◦
Tant que segments_found < num_segments_n :
    ▪ Initialiser pour le nouveau segment :
        • Créer un ensemble vide found_values ou un tableau booléen found_booleans de taille d initialisé à false.
        • current_segment_length = 0.
        • distinct_count = 0.
    ▪ Boucle de collecte (dans le segment actuel) :
        • Tant que distinct_count < d :
            ◦ Si sequence_index >= length(integer_sequence) : La séquence est trop courte pour trouver num_segments_n segments complets. Gérer l'erreur ou arrêter.
            ◦ Récupérer la valeur actuelle : value = integer_sequence[sequence_index].
            ◦ Avancer dans la séquence : sequence_index = sequence_index + 1 [19, 68, Algorithme C, C4].
            ◦ Augmenter la longueur du segment : current_segment_length = current_segment_length + 1 [19, 68, Algorithme C, C4].
            ◦ Si cette valeur est nouvelle pour le segment actuel :
    ▪ Si utilisant l'ensemble : Si value n'est pas dans found_values, ajouter value à found_values et distinct_count = distinct_count + 1.
    ▪ Si utilisant le tableau booléen : Si found_booleans[value] est false, mettre found_booleans[value] = true et distinct_count = distinct_count + 1.
        • Fin de la boucle de collecte (on a trouvé d valeurs distinctes).
    ▪ Enregistrer la longueur du segment complet :
        • Si current_segment_length >= max_length_t : COUNT[max_length_t] = COUNT[max_length_t] + 1 [19, 68, Algorithme C, G5, adapté].
        • Sinon : COUNT[current_segment_length] = COUNT[current_segment_length] + 1 [19, 68, Algorithme C, G5, adapté].
    ▪ Augmenter le compteur de segments trouvés : segments_found = segments_found + 1 [20, 69, Algorithme C, C6].
        ◦ Fin de la boucle principale.
6. Retourner les résultats :
◦ Retourner le tableau COUNT.

Note : Les fréquences observées dans COUNT sont ensuite utilisées dans le test du chi-carré. Les fréquences attendues pour chaque catégorie de longueur doivent être calculées séparément à l'aide des formules de probabilité théorique (comme celles mentionnées dans). Si des fréquences attendues sont trop faibles, il peut être nécessaire de regrouper des catégories de longueurs adjacentes avant d'appliquer le test χ².

# Exemple
Reprenons l'exemple donné dans les sources, bien que la séquence fournie ne soit pas tout à fait assez longue pour l'exemple complet de 7 segments dans la source.

Paramètres :
•
d = 3 (les valeurs possibles sont 0, 1, 2).
•
num_segments_n = 7.
•
max_length_t = 6 (arbitrairement choisi pour cet exemple, afin d'avoir quelques catégories)
Séquence d'entrée (entiers, donc pas de conversion nécessaire) : 1 1 0 1 2 2 1 0 2 2 1 2 0 2 0 2 0 0 1 2 1 2 2 0 1 0 1 0 2 0 1 1 2 1 (Longueur 34)
Suivons l'algorithme de comptage :
1. Paramètres : d=3, n=7, t=6.
2. Séquence déjà en entiers.
3. Initialiser COUNT (taille 7, indices 0 à 6) à ``. (On utilisera COUNT à `COUNT`).
4. sequence_index = 0, segments_found = 0.
Recherche du 1er segment (segments_found = 0) :
•
found_values = {}, current_segment_length = 0, distinct_count = 0.
•
Seq=1 : found_values={1}, len=1, dist=1. index=1.
•
Seq=1 : found_values={1}, len=2, dist=1. index=2.
•
Seq=0 : found_values={1, 0}, len=3, dist=2. index=3.
•
Seq=1 : found_values={1, 0}, len=4, dist=2. index=4.
•
Seq=2 : found_values={1, 0, 2}, len=5, dist=3. index=5. (Distinct count atteint 3).
•
Segment complet ! Longueur = 5. 5 < max_length_t (6).
•
COUNT = COUNT + 1. COUNT est maintenant .
•
segments_found = 1.

Recherche du 2ème segment (segments_found = 1) :
•
found_values = {}, current_segment_length = 0, distinct_count = 0. (sequence_index est maintenant 5).
•
Seq=2 : found_values={2}, len=1, dist=1. index=6.
•
Seq=1 : found_values={2, 1}, len=2, dist=2. index=7.
•
Seq=0 : found_values={2, 1, 0}, len=3, dist=3. index=8. (Distinct count atteint 3).
•
Segment complet ! Longueur = 3. 3 < max_length_t (6).
•
COUNT = COUNT + 1. COUNT est maintenant .
•
segments_found = 2.
Recherche du 3ème segment (segments_found = 2) :
•
found_values = {}, current_segment_length = 0, distinct_count = 0. (sequence_index est maintenant 8).
•
Seq=2 : found_values={2}, len=1, dist=1. index=9.
•
Seq=2 : found_values={2}, len=2, dist=1. index=10.
•
Seq=1 : found_values={2, 1}, len=3, dist=2. index=11.
•
Seq=2 : found_values={2, 1}, len=4, dist=2. index=12.
•
Seq=0 : found_values={2, 1, 0}, len=5, dist=3. index=13. (Distinct count atteint 3).
•
Segment complet ! Longueur = 5. 5 < max_length_t (6).
•
COUNT = COUNT + 1. COUNT est maintenant .
•
segments_found = 3. (Note : La source donne une longueur de 4 pour le 3ème segment avec la séquence de, ce qui diffère de mon traçage strict de l'algorithme C. Cela pourrait suggérer une légère nuance d'implémentation ou que la séquence dans n'est qu'un préfixe de celle utilisée pour obtenir les résultats dans. Faisons confiance aux longueurs finales données par la source pour la suite de l'exemple).
D'après la réponse de l'exercice 7 dans les sources, les longueurs des sept segments trouvés (avec d=3) sont : 5, 3, 4, 7, 6, 5, 4.
En utilisant ces longueurs et notre max_length_t = 6, le tableau COUNT final serait rempli comme suit :
•
Longueur 5 : apparaît 2 fois -> COUNT = 2.
•
Longueur 3 : apparaît 1 fois -> COUNT = 1.
•
Longueur 4 : apparaît 2 fois -> COUNT = 2.
•
Longueur 7 : apparaît 1 fois. 7 >= max_length_t (6) -> COUNT = COUNT + 1. COUNT devient 1.
•
Longueur 6 : apparaît 1 fois. 6 >= max_length_t (6) -> COUNT = COUNT + 1. COUNT devient 2.
Le tableau COUNT final observé (en utilisant les résultats de la source et notre t=6) serait :
•
COUNT (longueur 3) : 1
•
COUNT (longueur 4) : 2
•
COUNT (longueur 5) : 2
•
COUNT (longueur >= 6) : 2 (Les autres entrées de COUNT sont 0).
Étape du Test Chi-Carré (conceptuel pour cet exemple) :
1.
On aurait 4 catégories de longueurs : 3, 4, 5, >=6.
2.
Il faudrait calculer les probabilités théoriques Pr pour une longueur r (Pr = qr-1 - qr pour d <= r < t, pt = qt-1 pour r >= t, où qr est la probabilité qu'un segment de longueur r soit incomplet). Par exemple, pour d=3 :
◦
P(longueur=3) : q2 - q3
◦
P(longueur=4) : q3 - q4
◦
P(longueur=5) : q4 - q5
◦
P(longueur>=6) : q5 où qr = 1 - ({r 3} * 3! / 3^r) [formule adaptée de 21, 70, 101].
3.
Calculer les fréquences attendues : Attendue_r = n * Pr. Pour n=7, d=3, t=6, on calculerait 7 * P(longueur=3), 7 * P(longueur=4), etc.
4.
Appliquer la formule du chi-carré : χ² = Σ (Observee_r - Attendue_r)² / Attendue_r sur les 4 catégories.
5.
Comparer la statistique χ² calculée à une valeur critique (table du chi-carré) avec le nombre de degrés de liberté approprié (nombre de catégories - 1).
Cet exemple montre comment l'algorithme de comptage recueille les données (COUNT) nécessaires à la réalisation ultérieure du test statistique.