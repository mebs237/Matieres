"""
Module d'implementation des tests statistiques

Tests_classes
-----
    Les classes sont :

    - ``Tests`` : classe abstraite servant d'interface autres classes , les obligeant à implémenter la méthode ``test`` , fonction principale qui exécute le test statistique.
    - ``Chi2Test`` : test du **χ²** qui analyse les féquence d'apparition de éléments d'une séquence
    - ``GapTest`` : test du **Gap** qui analyse les gaps (sauts) entre les éléments d'une séquence
    - ``PokerTest`` : test du **Poker** qui analyse la fréquence d'apparition des motifs dans une séquence
    - ``CouponCollectorTest`` : test du **Collectionneur de Coupons** qui analyse
    - ``MaximumTest`` : test des **Maximum** qui analyse la répartition des maxima de groupes dans la séquence
    - ``KSTest`` : test de **Kolmogorov-Smirnov** analyse la différence entre la fonction de repartion théorique et celle observée

Fontions_Auxiliaires
--------------------
    Les fonctions auxilliaires à l'implémentation des test :

    -discretize : pour discretiser des séquences de nombres à virgules en séquence d'entiers
    -stirling : nombre de stirling de 2e espèce

Utilisation
-----------
    Pour une ``sequence`` , on teste l'hypothèse **H₀ : 'la séquence suit une distribution uniforme entre 0 et 1'** avec le test statistique T en :

    - *Instanciation* : créer une instance  ``t`` de la classe representant le test voulu , en fixant ou pas les paramètres du test ; si des paramètres ne sont pas spécifiés , les par défaut seront utilisés
    - *Application* : appliquer la méthode ``test`` sur ``sequence`` par ``t``
    - *Interpretation* : la valeur ``accept`` du resultat obtenu  si oui (``True``) ou non (``False``) on accepte l'hypothèse nulle H₀ (ou plus précisément , on ne rejette pas cette hypothése)

Exemple
-------

    >>> sequence = [5,9,3,1,4,9,6,2,3,4,...]
    >>> test = Chi2Test()
    >>> res_test = test.test(sequence , alpha = 0.01)
    >>> print(f"selon le test du χ² la séquence suit une loi uniforme ? {res_test['accept]} ")
    selon le test du χ² la séquence suit une loi uniforme ? True

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass , field
from collections import Counter
from typing import Dict , Union, Optional , Tuple , Callable
from functools import lru_cache
from math import factorial , comb
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 , kstwobign

@dataclass
class Result:
    """
    Classe pour tous les résultats de tests statistiques.
    """
    stat_obs: float = field(default=0.0)
    stat_crit: float = field(default=0.0)
    p_value: float = field(default=1.0)
    accept: bool = field(default=False)
    add_info: Optional[Dict] = field(default=None)


class Tests(ABC):
    """
    Classe abstraite de base pour tous les tests statistiques.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Retourne le nom du test."""

    @abstractmethod
    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False
            ) -> Result:
        """
        Exécute le test statistique sur les données.

        Args
        ---
            data : Séquence de nombres à tester
            alpha : Niveau de signification (défaut: 0.05)
            info : Si True, retourne ( result , add_info )
            **kwargs : paramètres additionnel dépendament du test implémenté

        Returns
        ---
            dic :
                un dictionnaire contenant :
            - stat_obs : float
                statistique observée/calculée
            - stat_crit : float
                valeur critique de la statistique
            - p_value : float
                p valeur
            - accept : bool
                on accepte l'hypothèse nulle 'H0' ; le test n'a pas detecté une différence significative ;
            - add_info : dict
                dictionnaire avec des information additionnelles sur le test
        """

    def _validate_input(self, data: NDArray) -> None:
        """Valide les données d'entrée."""
        if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype,np.number):
            raise TypeError("Les données doivent être un numpy.ndarray de nombres")
        if data.size == 0:
            raise ValueError("Les données ne peuvent pas être vides")
        if not np.all(np.isfinite(data)):
            raise ValueError("Les données doivent contenir uniquement des nombres finis")

    @staticmethod
    @lru_cache(maxsize=None)
    def stirling(n: int, k: int) -> int:
        """
        calcule le nombre de stirling de deuxième espèce.
        """
        if n < 0 or k < 0:
            raise ValueError("n et k doivent être des entiers positifs")

        if n == k:
            return 1

        elif n == 0 or k == 0:
            return 0

        return k * Tests.stirling(n - 1, k) + Tests.stirling(n - 1, k - 1)

    @staticmethod
    def discretize(data: NDArray ,
                a: Union[int , float] = 0,
                b : Union[int , float] = 1,
                n_bins:Optional[int] = None,
                binning : str = "quantile") -> Tuple[NDArray, int]:
        """
        Discrétisation des éléments d'une séquence entre ``[a,b]`` en ``n_bins`` catégories/classes selon la méthode de bining choisie.

        Parameters
        ----------
        data : séquence à diviser
        a : borne inférieure de l'intervalle des valeurs de data
        b : brone supérieure de l'intervalle des valeurs de data
        n_bins : nombre de catégories/intervalles désirés
        bining : méthode de division/discrétisation

        Fonctionnement
        --------------
        - si data est une liste d'entiers , le nombres de classes/catégories c'est le nombre de valeurs uniques
        - sinon ( c'est a dire contient des valeurs continues) , en supposant que ``[a,b]`` soit la plage de ces valeurs ;
            * on divise la plage en ``n_bins`` intervalles selon la méthode ``binning`` demandée par quantille ( binning = 'quantile' ) ou par intervalle fixe (binning = 'uniform')
            * à chaque valeur de data , on associe l'indice/index de l'intevalle auquel il appratient


        Returns
        -------
            discretize (NDArray):
                la séquence dicrétisée
                * si data est une séquence d'entiers : retourne la même séquence
                * sinon retourne une séquence où chaque élément est remplacé par l'index de la classe à laquel il appertient
            d (int):
                nombres de catégoris/classes
        """
        data = np.asarray(data)
        n = len(data)

        if np.issubdtype(data.dtype, np.integer):
            return data, len(np.unique(data))

        d = n_bins if not n_bins is None else int(1 + np.log2(len(data)))  # règle de Sturges
        d = max(1,d) # au moins 1 classe

        # normalisation des données entre a et b

        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

        data_norm = np.clip(data_norm , 0,1)

        if binning == "quantile":
            bins = np.quantile(data_norm, np.linspace(0,1,d + 1)[1:-1])

        elif binning == "uniform":
            bins = np.linspace(0 , 1 , d + 1)[1:-1]

        else:
            raise ValueError("méthodes de division ( ``binning``) non pris en charge , essayez 'quantile' ou 'uniform' ")

        discretized = np.digitize(data_norm, bins)
        if len(discretized) != n:
            raise ValueError("Erreur penandant la discretisation des données")
        return discretized, d


class Chi2Test(Tests):
    """
    Classe implémentant le Test du Χ²

    Args:
        k : Nombre de catégories (défaut: 10)
        min_expected : Fréquence minimale attendue pour au moins 80% des catégories (défaut: 5.0)
        probabilities : Probabilités théoriques pour chaque catégorie (défaut: None, distribution uniforme)

    """

    def __init__(self,
                 k: int = 10,
                 min_expected: float = 5.0,
                 probabilities: Optional[NDArray] = None)->None:
        """
        Initialise le test du Chi-carré.
        """
        self.k = k
        self.min_expected = min_expected
        self.probabilities = probabilities

    def __str__(self) -> str:
        return f"Chi2_Test(nb_bins = {self.k})"

    @staticmethod
    def group_low_frequence(observed: NDArray,
                        expected: NDArray,
                        min_expected: float = 5.0) -> Tuple[NDArray, NDArray]:
        """
        Regroupe en effectifs les classes d'effectif faibles en priorisant les classes adjacentes

        Args:
            observed: Liste des effectifs observés
            expected: Liste des effectifs attendus
            min_expected: Seuil principal pour les effectifs (défaut: 5)

        Returns:
            Tuple : (observed_new, expected_new) contenant les nouvelles listes regroupées
        """
        # Convertir en numpy arrays si nécessaire
        observed = np.array(observed)
        expected = np.array(expected)

        # Vérifier que les listes ont la même taille
        if len(observed) != len(expected):
            raise ValueError("Les listes observed et expected doivent avoir la même taille")

        # Vérifier que les listes ne sont pas vides
        if len(observed) == 0:
            return observed, expected

        # Copier les listes pour ne pas modifier les originales
        observed_new = observed.copy()
        expected_new = expected.copy()

        # Tant qu'il y a des classes avec effectif attendu < min_expected
        while np.any(expected_new < min_expected) and len(expected_new) > 1:
            # Trouver l'index de la classe avec l'effectif attendu minimal
            idx_min = np.argmin(expected_new)

            # Déterminer l'indice adjacent avec lequel regrouper
            if idx_min == 0:
                # Si c'est la première classe, regrouper avec la suivante
                idx_adjacent = 1
            elif idx_min == len(expected_new) - 1:
                # Si c'est la dernière classe, regrouper avec la précédente
                idx_adjacent = idx_min - 1
            else:
                # Sinon, regrouper avec la classe adjacente ayant l'effectif attendu minimal
                if expected_new[idx_min-1] <= expected_new[idx_min+1]:
                    idx_adjacent = idx_min - 1
                else:
                    idx_adjacent = idx_min + 1

            # Regrouper les classes à gauche
            observed_new[min(idx_min, idx_adjacent)] += observed_new[max(idx_min, idx_adjacent)]
            expected_new[min(idx_min, idx_adjacent)] += expected_new[max(idx_min, idx_adjacent)]

            # Supprimer la classe regroupée
            observed_new = np.delete(observed_new, max(idx_min, idx_adjacent))
            expected_new = np.delete(expected_new, max(idx_min, idx_adjacent))

        return observed_new, expected_new

    @staticmethod
    def get_counts(data: NDArray,
                        k: int = 10,
                        probabilities: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
        """
        effectue les comptages d'effectifs selon chacune des ``k`` classes

        Parameters
        ---
            data:
                Données à regrouper en bins
            k:
                Nombre de classes/bins souhaité (défaut: 10)
            probabilities:
                Probabilités théoriques pour chaque classe (défaut: None, distribution uniforme)

        Returns
        ---
            Tuple :
                tuple contenant ;
            - observed: NDArray
                    Effectifs observés pour chaque classe/bin
            - expected: NDArray
                    Effectifs attendus pour chaque classe/bin
        """
        # Convertir en numpy array si nécessaire
        data = np.array(data)
        n = len(data)

        if n == 0:
            raise ValueError("Les données ne peuvent pas être vides")

        if np.issubdtype(data.dtype, np.integer):
            # Pour les entiers, utiliser les valeurs uniques comme classes
            unique_values = np.unique(data)
            d = len(unique_values)
            observed = np.bincount(data, minlength=max(data) + 1)[unique_values]
            if probabilities is None:
                # distribution uniforme
                expected = np.full(d, n / d)
            else:
                # Utiliser les probabilités fournies
                probabilities = np.array(probabilities)
                if len(probabilities) != d:
                   raise ValueError("Le nombre de probabilités doit correspondre au nombre de valeurs uniques")
                expected = n * probabilities
            return observed, expected


        observed , _ = np.histogram(data, bins=k)
        if probabilities is None:
            # Distribution uniforme
            expected = np.full(k, n / k)
        else:
            # Utiliser les probabilités fournies
            probabilities = np.array(probabilities)
            if len(probabilities) != k:
                raise ValueError("Le nombre de probabilités doit correspondre au nombre de classes")
            expected = n * probabilities

        return observed, expected

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Result:

        self._validate_input(data)

        # Calculer les fréquences observées et attendues
        observed, expected = self.get_counts(data, k=self.k, probabilities=self.probabilities)

        # Regrouper les catégories si nécessaire
        observed, expected = self.group_low_frequence(observed, expected, min_expected=self.min_expected)

        # Calculer la statistique du test
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - chi2.cdf(chi2_stat, df)
        critical_value = chi2.ppf(1 - alpha, df)

        result = Result(stat_obs = chi2_stat,
                        stat_crit = critical_value,
                        p_value = p_value,
                        accept = p_value > alpha )

        if info :
            result.add_info = {"df" : df ,
                        "observed" : observed,
                        "expected" : expected
                       }

        return result

    @staticmethod
    def test2(observed: NDArray,
              expected: NDArray,
              alpha: float = 0.05,
              info: bool = False
              ) -> Result:
        """
        Exécute le test du Chi-carré à partir des effectifs observés et attendus.

        Args:
            observed: Liste des effectifs observés
            expected: Liste des effectifs attendus
            alpha: Niveau de signification (défaut: 0.05)
            info: Si True, retourne (result, add_info)


        Returns:
            Résultat du test ou tuple (résultat, informations additionnelles)
        """
        # Vérifier les entrées
        if len(observed) != len(expected):
            raise ValueError("Les listes observed et expected doivent avoir la même taille")

        # regrouper les classes si nécessaire
        observed , expected = Chi2Test.group_low_frequence(observed=observed , expected=expected)

        # Calculer la statistique du test
        chi2_stat = np.nansum((observed - expected) ** 2 / expected)

        # Degrés de liberté
        df = len(observed) - 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - chi2.cdf(chi2_stat, df)
        critical_value = chi2.ppf(1 - alpha, df)

        result = Result(stat_obs = chi2_stat,
                        stat_crit = critical_value,
                        p_value = p_value,
                        accept =  p_value > alpha )

        if info:
            result.add_info = {"df": df,
                            "observed": observed,
                            "expected": expected
                            }

        return result


class GapTest(Tests):
    """
    Test du Gap.

    Args:
        alpha_gap: Borne inférieure de l'intervalle (défaut: 0.1)
        beta_gap: Borne supérieure de l'intervalle (défaut: 0.3)
        max_gap: Longueur maximale de gap à considérer (défaut: 30)
    """

    def __init__(self,
                 alpha_gap: float = 0.1,
                 beta_gap: float = 0.4,
                 max_gap: int = 30)->None:
        """
        Initialise le test du Gap.

        """
        if not 0 <= alpha_gap < beta_gap <= 1:
            raise ValueError("Les bornes doivent satisfaire 0 <= alpha_gap < beta_gap <= 1")

        self.alpha_gap = alpha_gap
        self.beta_gap = beta_gap
        self.max_gap = max_gap
        self.p = beta_gap - alpha_gap # probabilité d'être marqué

    def __str__(self) -> str:
        return f"Gap_Test( intervall =  [ {self.alpha_gap} , {self.beta_gap} ] )"

    def _compute_gap_probabilities(self, max_gap: int) -> NDArray:
        """
        Calcule les probabilités théoriques pour chaque longueur de gap
        """

        q = 1 - self.p
        probs = np.zeros(max_gap + 1)
        probs[:-1] = self.p * (q ** np.arange(max_gap))
        probs[-1] = q ** max_gap
        return probs

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Result:
        """
        Exécute le test du Gap.
        """
        self._validate_input(data)

        max_data = np.max(data)
        min_data = np.min(data)
        # Vérifier que les données sont dans [0, 1] ; sinon , normaliser les données
        if min_data < 0 or max_data > 1:
            data = (data-min_data)/(max_data-min_data)

        # Identifier les nombres dans l'intervalle [alpha_gap, beta_gap]
        marks = (data >= self.alpha_gap) & (data <= self.beta_gap)
        indices = np.where(marks)[0]

        if len(indices) <= 1:
            return Result(add_info={"message": "Pas assez de marques pour calculer des gaps"})


        # Calculer les longueurs des gaps
        gaps = np.diff(indices) - 1
        max_observed_gap = np.max(gaps)
        # ajuster la limite maximal de longueur de gap
        t = min(max_observed_gap, self.max_gap)

        # Compter les occurrences de chaque longueur de gap
        gap_counts = np.zeros(t + 1)
        for gap in gaps:
            if gap >= t:
                gap_counts[t] += 1
            else:
                gap_counts[gap] += 1

        # Calculer les probabilités théoriques
        expected_probs = self._compute_gap_probabilities(t)
        expected_counts = len(gaps) * expected_probs

        # Appliquer le test du chi-carré

        result =  Chi2Test.test2(gap_counts,expected_counts, alpha=alpha,info=info)

        if info :
            result.add_info.update({"max_gap":t})
        return result


class PokerTest(Tests):
    """
    Test du Poker généralisé pour t>=2

    Attributes
    ----------
    group_size : int
        taille des groupes
    n_bins : int
        nombre de catégories pour la division

    """

    def __init__(self, group_size: int = 5,n_bins=None) -> None:
        """
        initialisation du test
        """
        if group_size < 2:
            raise ValueError("La taille des groupes doit être ≥ 2")

        self.group_size = group_size
        self.n_bins = n_bins

    def __str__(self) -> str:
        return f"Poker_Test( group_size = {self.group_size})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False)->Result:
        """
        Execute le test du poker
        """
        self._validate_input(data)
        n = len(data)

        # Discrétisation si flottants
        data_new , d = Tests.discretize(data=data,n_bins=self.n_bins)

        # Découpage en groupes
        n_group = n // self.group_size
        if n_group < 1:
            raise ValueError(f"Pas assez de données pour former des groupes de taille  {self.group_size} ")

        groups = data_new[:n_group * self.group_size].reshape(n_group, self.group_size)

        # Compter les motifs (nombres de valeurs distinctes par groupe)
        distinct_values = [len(set(group)) for group in groups]

        # nombre de catégories possibles (pour le test du chi-carré)
        d = min(self.group_size , d)

        # Compter  les groupes ayant r valeurs distinctes
        observed = np.zeros(d+1)  # +1 car on compte de 0 à d valeurs distinctes
        for r in distinct_values:
            observed[r] += 1

        # Calculer les probabilités théoriques pour chaque nombre r de valeurs distinctes
        probs = np.zeros(d+1)
        for r in range(1, d+1):
            probs[r] = Tests.stirling(self.group_size, r)*factorial(d)/(factorial(d-r)*(d**self.group_size))


        probs = probs[1:]  # Enlever le cas r=0 qui est impossible
        observed = observed[1:]  # Enlever le cas r=0 qui est impossible

        # Appliquer le test du chi2
        result = Chi2Test.test2(observed, n_group * probs, alpha=alpha, info=info)

        if info:
            result.add_info.update({
                "group_size": self.group_size,
                "n_bins": self.n_bins,
                "theoretical_probs": probs,
                "observed_counts": observed,
                "distinct_values_distribution": dict(Counter(distinct_values))
            })
        return result


class CouponCollectorTest(Tests):
    """
    Test du Collectionneur de Coupons.

    Attributes
    ----------
    d : int
        Nombre coupons , ou de valeurs distinctes possibles (si None, utilise le nombre de valeurs uniques)
    max_seg : int
        longueur maximal des segments à considérer
    """
    def __init__(self, d: Optional[int] = None,max_lenght:int = 30):
        """
        Initialise le test du Collectionneur de Coupons.

        """
        self.d = d
        self.max_lenght = max_lenght

    def __str__(self) -> str:
        return f"Coupon_Collector_Test (nb_coupon = {self.d})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Result:
        """
        Exécute le test du Collectionneur de Coupons.
        """
        self._validate_input(data)



        # Discrétisation
        data_new , d  = Tests.discretize(data=data , n_bins = self.d)

        t = self.max_lenght

        #-------------------------------------
        # Calculer des longueurs de segments
        #-------------------------------------


        len_segments = []
        current_segment = set()
        lenght = 0
        for value in data_new:
            current_segment.add(value)
            lenght += 1
            # si le segment est trop long ,  on le coupe et on le stocke
            if lenght >=t:
                current_segment = set()
                lenght = 0
                len_segments.append(t)
            # Si le segment est complet (contient toutes les valeurs distinctes)
            if len(current_segment) == d:
                len_segments.append(lenght)
                current_segment = set()
                lenght = 0

        if not len_segments: # aucun segment complet trouvé
            result = Result()

            if info :
                result.add_info = {"message": "Pas de segments complets trouvés"}

            return result

        len_segments = np.array(len_segments)
        #--------------------------------------------------------
        # Calcul des probabilités théoriques pour chaque longueur
        #--------------------------------------------------------
        size = t+1-d
        probs = np.zeros(size)
        for i in range(size-1):
            r = d+i # longueur de segment
            probs[i] = (factorial(d) / (d ** r)) * Tests.stirling(r - 1, d - 1)

        probs[size-1] = 1- np.sum(probs[:-1]) # dernier segment
        observed_counts = np.zeros(size)
        for val in len_segments:
            idx = val - d
            if 0<=idx<size:
                observed_counts[idx]+=1

        expected = len_segments.size * probs
        #---------------------------------
        # Appliquer le test du chi-carré sur la séquence des longueurs des segments
        #---------------------------------

        result = Chi2Test.test2(observed=observed_counts,expected=expected, alpha=alpha, info=info)

        if info :
            result.add_info.update( {
                "length_segments":len_segments ,
                "d": d,
                "max_lenght":t})

        return result


class KSTest(Tests):
    """
    Test de Kolmogorov-Smirnov pour comparer la distribution empirique
    à une distribution théorique.

    Attributes
    ----------
    cdf : callable
        Fonction de répartition théorique. Par défaut, fonction de repartition uniforme sur [0,1]
    """

    def __init__(self,
                 cdf: Optional[Callable[[NDArray], NDArray]] = None):
        """
        Initialise le test de Kolmogorov-Smirnov.
        """
        # Fonction de répartition par défaut (loi uniforme sur [0,1])
        def uniform_cdf(x: NDArray) -> NDArray:
            return np.clip(x, 0, 1)

        self.cdf = cdf if cdf is not None else uniform_cdf

    def __str__(self) -> str:
        return "K-S_Test"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False
             ) -> Result:
        """
        Exécute le test de Kolmogorov-Smirnov.
        """

        self._validate_input(data)

        # Trier les données
        sorted_data = np.sort(data)
        n = len(data)

        # Calculer la fonction de répartition empirique
        Fn = (np.arange(n) + 1 )/n
        Fm = (np.arange(n))/n

        # Calculer la fonction de répartition théorique
        Ft = self.cdf(sorted_data)

        # Calculer les différences maximales
        D_plus = np.max(Fn - Ft)
        D_minus = np.max(Ft - Fm)
        D = max(D_plus, D_minus)

        # Calculer la statistique de test
        ks_stat = D * np.sqrt(n)

        # Calculer la p-value et la valeur critique
        p_value = 1 - kstwobign.cdf(ks_stat)
        critical_value = kstwobign.ppf(1 - alpha)

        result = Result(stat_obs = ks_stat,
                        stat_crit = critical_value,
                        p_value = p_value,
                        accept = p_value > alpha)

        if info:
            result.add_info = {'ks_df': n,
                            "D_plus": D_plus,
                            "D_minus": D_minus,
                            "n": n
                            }
        return result


class MaximumTest(Tests):
    """
    Test des maximums pour analyser la repartition des maximums dans une séquence aléatoire supposée uniforme.

    Attributes
    ----------
    t : int
        taille de groupe à considérer (défaut: 5)
    """

    def __init__(self, t: int = 5):
        """
            Initialise le test des maximums.
        """
        if t < 2:
            raise ValueError("t doit être au moins 2")
        self.t = t

    def __str__(self) -> str:
        return f"Maximum_Test ( group_size = {self.t} )"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Result:

        self._validate_input(data)

        n = len(data)
        if n < self.t:
            raise ValueError(f"La taille de l'échantillon ({n}) doit être au moins égale à t ({self.t})")

        # Diviser les données en groupes de taille t

        groups = [data[i:i + self.t] for i in range(0 , n - self.t + 1 , self.t)]

        # Calculer les maximums de chaque groupe
        max_values = np.array([np.max(group) for group in groups])

        def max_cdf(x):
            """fonction de repartition des maximum """
            return x**self.t

        #----------------------------------------
        # Appliquer le test de Kolmogorov-Smirnov
        #----------------------------------------

        kstest = KSTest(cdf = max_cdf)
        result = kstest.test(data = max_values ,
                               alpha=alpha ,
                               info=info)
        if info :
            result.add_info.update({'max_df':self.t-1,
                              "max_values": max_values,
                              "t": self.t
                              })

        return result
