# Explication du Test du Poker (Test de Partition)
Le test du poker, également appelé test de partition, est un test statistique utilisé pour évaluer le caractère aléatoire d'une séquence de nombres, particulièrement adapté pour vérifier si les nombres générés se comportent comme s'ils étaient tirés indépendamment d'une distribution uniforme. L'idée sous-jacente est similaire au jeu de poker avec des dés.
Le test consiste à examiner des groupes de nombres consécutifs (appelés k-tuples) issus de la séquence testée. Pour chaque k-tuple, on observe le "motif" ou la configuration des valeurs qu'il contient. Ces motifs sont ensuite regroupés en différentes "catégories" (ou "classes") basées sur le type de configuration. Enfin, un test du chi-carré (χ²) est appliqué pour comparer la fréquence observée de chaque catégorie avec la fréquence théoriquement attendue pour une séquence réellement aléatoire.

# Distinction entre Motifs, Catégories, Classes et Valeurs distinctes

* Motifs / Patterns: Dans le contexte du test du poker, un "motif" ou "pattern" est la configuration spécifique des valeurs dans un k-tuple. Par exemple, avec k=5 nombres, des motifs pourraient être "tous différents", "une paire", "deux paires", "trois d'une sorte", "full", "quatre d'une sorte", "cinq d'une sorte". Ces termes correspondent aux mains du jeu de poker.

* Catégories / Classes: Les "catégories" ou "classes" sont les groupes prédéfinis dans lesquels les motifs observés sont classifiés pour le test du chi-carré. Il peut s'agir des motifs individuels (par exemple, chaque type de main de poker est une catégorie distincte). Les sources mentionnent spécifiquement sept patterns (les mains de poker pour k=5) comme catégories.

* Valeurs distinctes / Cases différentes: Une version "plus simple" du test du poker, mentionnée dans les sources, regroupe les motifs non pas par le type de main de poker exacte, mais par le nombre de valeurs distinctes présentes dans le k-tuple. Par exemple, pour k=5 et d=6 (comme des dés à 6 faces), les motifs sont regroupés en fonction du nombre de valeurs différentes qu'ils contiennent : 1 valeur distincte (cinq identiques), 2 valeurs distinctes (quatre identiques + une différente, ou trois identiques + deux identiques), 3 valeurs distinctes (trois identiques + deux différentes, ou deux identiques + deux identiques + une différente), 4 valeurs distinctes (une paire + trois différentes), 5 valeurs distinctes (cinq différentes). Dans cette version simplifiée, les catégories pour le test du chi-carré sont définies par le nombre de valeurs distinctes (1, 2, 3, 4, ou 5 dans cet exemple).

En résumé, les "motifs" sont les configurations brutes observées. Les "catégories" ou "classes" sont les groupes dans lesquels ces motifs sont classés pour le test statistique ultérieur. Le "nombre de valeurs distinctes" est un critère spécifique utilisé dans une version simplifiée du test du poker pour définir ces catégories.

# Application à une Séquence de Nombres Réels vs. Entiers
Le test du poker tel que décrit dans les sources s'applique typiquement à une séquence de nombres entiers. Cependant, ces séquences d'entiers (notées Yj) sont souvent dérivées d'une séquence de nombres réels (notées Uj) qui prétendent être indépendamment et uniformément distribués entre zéro et un.
La conversion d'une séquence de réels Uj en une séquence d'entiers Yj est expliquée dans les sources: on choisit un nombre d, et Yj est défini comme la partie entière de d * Uj (Yj = Ld * UjJ). Cette règle mappe chaque nombre réel Uj (dans [0, 1[) à un entier Yj (dans [0, d-1]). Cela revient à diviser l'intervalle [0, 1[ en d sous-intervalles égaux et à attribuer à chaque nombre réel l'indice de l'intervalle dans lequel il tombe. Le test du poker est ensuite appliqué à la séquence d'entiers Yj ainsi obtenue. Le choix de d est important: il doit être assez grand pour que le test soit significatif, mais pas trop grand pour ne pas rendre le test impraticable.

# Algorithme Simple pour le Test du Poker (Version Simplifiée basée sur le nombre de valeurs distinctes)
Cet algorithme calcule la statistique du test du chi-carré pour le test du poker simplifié (basé sur le nombre de valeurs distinctes), pour un échantillon de n k-tuples, où les valeurs proviennent d'une distribution discrète uniforme sur d valeurs possibles (0 à d-1).
1. Choisir les paramètres:
◦
k: La taille de chaque groupe (k-tuple). Typiquement 5.
◦
d: Le nombre de valeurs discrètes possibles. Pour des nombres réels dans [0, 1[, c'est le nombre d'intervalles dans lequel on divise [0, 1[.
◦
n: Le nombre de k-tuples à examiner. (Si vous avez une longue séquence de longueur totale N, n = N / k).
2. Définir les catégories et calculer les probabilités attendues:
◦
Les catégories sont basées sur le nombre r de valeurs distinctes présentes dans un k-tuple. Il y a min(k, d) catégories possibles, correspondant à 1, 2, ..., min(k, d) valeurs distinctes.
◦
Pour chaque catégorie r (de 1 à min(k, d)), calculez la probabilité théorique Pr qu'un k-tuple généré aléatoirement contienne exactement r valeurs distinctes. La source donne une formule (à interpréter avec prudence pour {k r} qui est probablement un nombre de Stirling de seconde espèce): Pr = ({k r} * d * (d-1) * ... * (d-r+1)) / d^k. Cette formule peut aussi s'écrire Pr = {k r} * d! / ((d-r)! * d^k).
◦
Calculez les fréquences attendues pour chaque catégorie r: Attendue_r = n * Pr.
3. Initialiser les compteurs: Créez un tableau ou une liste pour stocker le nombre observé de k-tuples pour chaque catégorie r (de 1 à min(k, d)). Initialisez tous les compteurs à 0.
4. Traiter la séquence de données:
◦
Si vous avez une séquence de nombres réels Uj, convertissez-la d'abord en une séquence d'entiers Yj = Ld * UjJ.
◦
Divisez la séquence d'entiers (ou d'entiers convertis) en n k-tuples consécutifs. Pour j allant de 0 à n-1, considérez le k-tuple (Y_j*k, Y_j*k+1, ..., Y_j*k+k-1).
5. Pour chaque k-tuple:
◦
Identifiez les valeurs uniques présentes dans ce k-tuple.
◦
Comptez le nombre r de ces valeurs uniques.
◦
Incrémentez le compteur correspondant à la catégorie r.
6. Calculer la statistique du chi-carré:
◦
Initialisez chi2_stat = 0.
◦
Pour chaque catégorie r (de 1 à min(k, d)):
▪
Observez le nombre Observee_r dans le compteur correspondant.
▪
Si Attendue_r > 0 (pour éviter la division par zéro et les catégories vides), calculez terme = (Observee_r - Attendue_r)² / Attendue_r.
▪
Ajoutez terme à chi2_stat.
◦
Note: Si certaines catégories attendues sont trop petites (par exemple, moins de 5), il est courant de regrouper ces catégories adjacentes pour que le test chi-carré soit valide. Cela réduit le nombre de degrés de liberté.
7. Prendre une décision:
◦
Le nombre de degrés de liberté pour le test du chi-carré est le nombre de catégories réellement utilisées (après un éventuel regroupement) moins 1.
◦
Comparez la valeur chi2_stat calculée à une valeur critique tirée d'une table de distribution du chi-carré pour le niveau de signification α souhaité et le nombre de degrés de liberté calculé.
◦
Si chi2_stat est supérieur à la valeur critique, vous rejetez l'hypothèse nulle selon laquelle la séquence est uniformément distribuée (pour ce test particulier).

# Exemple Simple
Simplifions l'exemple pour le rendre facile à suivre à la main, même s'il ne serait pas significatif statistiquement avec si peu de données.
Supposons que nous voulions tester une courte séquence de nombres réels pour le caractère aléatoire. Paramètres:
•
k = 2 (on regarde des paires)
•
d = 4 (on va diviser [0, 1[ en 4 intervalles: [0, 0.25[, [0.25, 0.5[, [0.5, 0.75[, [0.75, 1[ -> valeurs entières 0, 1, 2, 3)
•
Séquence de réels: {0.1, 0.6, 0.4, 0.9, 0.3, 0.7, 0.2, 0.8}
•
Taille de la séquence N = 8. Nombre de k-tuples n = N / k = 8 / 2 = 4.
Suivons l'algorithme:
1.
Paramètres: k=2, d=4, n=4.
2.
Catégories et Probabilités attendues (pour k=2, d=4):
◦
Catégorie r=1 (1 valeur distincte, ex: (0,0), (1,1)): Pr = {2 1} * 4! / ((4-1)! * 4²) = 1 * 4 / (6 * 16) = 4/16 = 1/4 = 0.25.
◦
Catégorie r=2 (2 valeurs distinctes, ex: (0,1), (3,2)): Pr = {2 2} * 4! / ((4-2)! * 4²) = 1 * (4*3) / (2 * 16) = 12/16 = 3/4 = 0.75.
◦
Fréquences attendues (n=4):
▪
Attendue_1 = 4 * 0.25 = 1.
▪
Attendue_2 = 4 * 0.75 = 3.
3.
Initialiser compteurs: Observee_1 = 0, Observee_2 = 0.
4.
Convertir les réels en entiers (Yj = L4 * UjJ):
◦
0.1 -> L4*0.1J = L0.4J = 0
◦
0.6 -> L4*0.6J = L2.4J = 2
◦
0.4 -> L4*0.4J = L1.6J = 1
◦
0.9 -> L4*0.9J = L3.6J = 3
◦
0.3 -> L4*0.3J = L1.2J = 1
◦
0.7 -> L4*0.7J = L2.8J = 2
◦
0.2 -> L4*0.2J = L0.8J = 0
◦
0.8 -> L4*0.8J = L3.2J = 3 Séquence d'entiers Y: {0, 2, 1, 3, 1, 2, 0, 3}.
5.
Traiter les k-tuples (paires):
◦
K-tuple 1: (Y0, Y1) = (0, 2). Valeurs distinctes: {0, 2}. Nombre distinctes: 2. Incrémenter Observee_2. Observee_2 = 1.
◦
K-tuple 2: (Y2, Y3) = (1, 3). Valeurs distinctes: {1, 3}. Nombre distinctes: 2. Incrémenter Observee_2. Observee_2 = 2.
◦
K-tuple 3: (Y4, Y5) = (1, 2). Valeurs distinctes: {1, 2}. Nombre distinctes: 2. Incrémenter Observee_2. Observee_2 = 3.
◦
K-tuple 4: (Y6, Y7) = (0, 3). Valeurs distinctes: {0, 3}. Nombre distinctes: 2. Incrémenter Observee_2. Observee_2 = 4. Compteurs finaux: Observee_1 = 0, Observee_2 = 4.
6.
Calculer la statistique du chi-carré:
◦
Catégorie r=1: Observee=0, Attendue=1. Terme = (0 - 1)² / 1 = 1 / 1 = 1.
◦
Catégorie r=2: Observee=4, Attendue=3. Terme = (4 - 3)² / 3 = 1 / 3 ≈ 0.333.
◦
chi2_stat = 1 + 0.333 ≈ 1.333.
7.
Décision:
◦
Nombre de catégories utilisées = 2. Degrés de liberté = 2 - 1 = 1.
◦
On comparerait chi2_stat = 1.333 à une valeur critique d'une table du chi-carré avec 1 degré de liberté. Pour un niveau α=0.05, la valeur critique est d'environ 3.841.
◦
Comme 1.333 < 3.841, on ne rejette pas l'hypothèse nulle. La séquence semble compatible avec une distribution uniforme selon ce petit test.
Cet exemple illustre le calcul. En pratique, on utilise des échantillons n beaucoup plus grands pour que le test soit valide et significatif. Le regroupement des catégories peut être nécessaire si les fréquences attendues sont trop faibles.