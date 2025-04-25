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
from typing import Dict , Union, Optional , Tuple , Any
from functools import lru_cache
from collections import Counter
from math import factorial , comb
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 , kstwobign , kendalltau , norm
from Utils import (stirling ,
                   get_optimal_bins ,
                   group_low_frequence ,
                   generate_motifs ,
                   assign_motifs)

class Tests(ABC):
    """Classe abstraite de base pour tous les tests statistiques."""

    @abstractmethod
    def __str__(self) -> str:
        """Retourne le nom du test."""

    @abstractmethod
    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test statistique sur les données.

        Args
        ---
            data: Séquence de nombres à tester
            alpha: Niveau de signification (défaut: 0.05)
            info: Si True, retourne ( result , add_info )

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
            - violation : bool
                True si stat_obs > stat_crit
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
        return f"Chi2Test(k={self.k})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:

        self._validate_input(data)

        # Calculer les fréquences observées et attendues
        observed, expected = get_optimal_bins(data, k=self.k, probabilities=self.probabilities)

        # Regrouper les catégories si nécessaire
        observed, expected = group_low_frequence(observed, expected, min_expected=self.min_expected)

        # Calculer la statistique du test
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - chi2.cdf(chi2_stat, df)
        critical_value = chi2.ppf(1 - alpha, df)

        result = {'stat_obs': chi2_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value > alpha,
                  'violation': chi2_stat > critical_value
                 }

        if info :
            add_info = {"df" : df ,
                        "observed" : observed,
                        "expected" : expected
                       }
            return result , add_info
        else :
            return result


def chi2_from_other(observed: NDArray,
                    expected: Optional[NDArray] = None, probabilities: Optional[NDArray] = None,
                    alpha: float = 0.05,
                    min_expected : float =5.0,
                    info: bool = False,
                    extra_info: Optional[Dict] = None) -> Union[Dict, Tuple[Dict]]:
    """
    Applique le test du Chi² à partir d'effectifs observés (et soit des effectifs attendus soit des probabilités).

    Args:
        observed: Effectifs observés
        expected: Effectifs attendus (optionnel si probabilities est fourni)
        probabilities: Probabilités théoriques à utiliser pour calculer expected
        alpha: Niveau de signification
        info: Retourner les détails ou non
        extra_info: Dictionnaire d'infos supplémentaires à inclure dans add_info

    Returns:
        Dictionnaire avec les résultats du test, et si info=True, un tuple (result, add_info)
    """
    observed = np.array(observed)
    n = np.sum(observed)

    # Calcul des effectifs attendus
    if expected is None:
        if probabilities is None:
            expected = np.full_like(observed, n / len(observed))
        else:
            probabilities = np.array(probabilities)
            if len(probabilities) != len(observed):
                raise ValueError("Probabilités et effectifs observés doivent avoir la même taille")
            expected = n * probabilities
    else:
        expected = np.array(expected)
        if len(expected) != len(observed):
            raise ValueError("Effectifs observés et attendus doivent avoir la même taille")

    # Regroupement si nécessaire
    observed, expected = group_low_frequence(observed, expected, min_expected=min_expected)

    # Statistique du test
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)

    result = {
        "stat_obs": chi2_stat,
        "stat_crit": critical_value,
        "p_value": p_value,
        "accept": p_value > alpha,
        "violation": chi2_stat > critical_value
    }

    if info:
        add_info = {
            "df": df,
            "observed": observed,
            "expected": expected
        }
        if extra_info:
            add_info.update(extra_info)
        return result, add_info
    else:
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
                 beta_gap: float = 0.3,
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
        return f"GapTest( [ {self.alpha_gap} , {self.beta_gap} ] )"

    def _compute_gap_probabilities(self, max_gap: int) -> NDArray:
        """Calcule les probabilités théoriques pour chaque longueur de gap"""

        q = 1 - self.p
        probs = np.zeros(max_gap + 1)
        probs[:-1] = self.p * (q ** np.arange(max_gap))
        probs[-1] = q ** max_gap
        return probs

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test du Gap.
        """
        self._validate_input(data)

        # Vérifier que les données sont dans [0, 1]
        if np.min(data) < 0 or np.max(data) > 1:
            raise ValueError("Les données doivent être dans l'intervalle [0, 1]")

        # Identifier les nombres dans l'intervalle [alpha_gap, beta_gap]
        marks = (data >= self.alpha_gap) & (data <= self.beta_gap)
        indices = np.where(marks)[0]

        if len(indices) <= 1:
            result = {
                'stat_obs':0,
                'stat_crit':0,
                'p_value':1.0,
                'accept' : False,
                'violation': False
            }

            add_info={"message": "Pas assez de marques pour calculer des gaps"}
            if info :
                return result , add_info
            else :
                return result

        # Calculer les gaps
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

        return chi2_from_other(gap_counts,expected_counts, alpha=alpha,info=info,extra_info={"max_gap":t})


class PokerTest(Tests):
    """
    Test du Poker généralisé pour t>=2

    Attributes:
        group_size (int): taille des groupes

    """

    def __init__(self, group_size: int = 5) -> None:
        if group_size < 2:
            raise ValueError("La taille des groupes doit être ≥ 2")

        self.group_size = group_size
        self._motifs = generate_motifs(group_size)

    def __str__(self) -> str:
        return f"PokerTest(t={self.group_size})"

    @lru_cache(maxsize=None)
    @staticmethod
    def motif_probabilities(t, d , motifs):
        """
        probabilité d'apparition théorique de chaque  motif

        Args
        ---
        t : taille du groupe
        d : nombre de catégories ou taille de l'alphabet
        motifs : ensemble des motifs possibles

        Returns
        -------
        dict :
            dictionnaire {nom_motif: probabilité théorique}


        """
        probs = {}
        for name , cfg in motifs:
            r = sum(cfg.values()) # nombres d'éléments distincts dans le motif
            probs[name] = stirling(t , r)*comb(d-r)*factorial(r)/d**t

        return probs

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False ) -> Union[Dict[str , Any] ,Tuple[Dict[str, Any], Dict[str, Any]]]:

        self._validate_input(data)
        n = len(data)
        # Discrétisation si flotants par la méthode des quantile
        if not np.issubdtype(data.dtype, np.integer):
            # choix de d idéal
            d = min(self.group_size , (1+np.log10(n)))
            bins = np.quantile(data, np.linspace(0, 1, d + 1)[1:-1])
            data = np.digitize(data, bins)  # indices 0..d-1 :contentReference[oaicite:8]{index=8}
        d = len(np.unique(data))

        # Découpage de la sequence en groupes
        n_group = n // self.group_size
        if n <= 2:
            raise ValueError(f"Pas assez de données pour {self.group_size}")

        groups = data[: n_group * self.group_size].reshape(n_group, self.group_size)

        # Attribution de motifs
        obs_counts: Dict[str, int] = {}
        for grp in groups:
            name = assign_motifs(grp.tolist(), self._motifs)
            obs_counts[name] = obs_counts.get(name, 0) + 1

        observed = np.array(obs_counts.values())

        # Probabilités théoriques de chaque motif
        probs = PokerTest.motif_probabilities(self.group_size , d, self._motifs)

        exp_probs = np.array([probs[name] for name in obs_counts])


        # test du χ²
        return chi2_from_other(observed=observed ,
                               probabilities=exp_probs ,
                               alpha=alpha, info=info,
                               extra_info={"motifs":self._motifs})


class CouponCollectorTest(Tests):
    """
    Test du Collectionneur de Coupons amélioré.

    Args:
        d: Nombre coupons , ou de valeurs distinctes possibles (si None, utilise le nombre de valeurs uniques)
        max_seg: longueur maximal des segments à considérer
    """

    def __init__(self, d: Optional[int] = None,max_lenght:int = 30):
        """
        Initialise le test du Collectionneur de Coupons.

        """
        self.d = d
        self.max_lenght = max_lenght

    def __str__(self) -> str:
        return f"CouponCollectorTest (d={self.d})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test du Collectionneur de Coupons.
        """
        self._validate_input(data)

        #
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Les données doivent être des entiers")

        # Déterminer le nombre de valeurs distinctes
        d = self.d if self.d is not None else len(np.unique(data))

        # Calculer les longueurs des segments
        segments = []
        current_segment = set()
        segment_length = 0

        for value in data:
            current_segment.add(value)
            segment_length += 1

            if len(current_segment) == d:
                segments.append(segment_length)
                current_segment = set()
                segment_length = 0

        if not segments:
            result = {'stat_obs': 0,
                  'stat_crit': 0,
                  'p_value': 1,
                  'accept': False,
                  'violation': False
                  }

            if info :
                add_info={"message": "Pas de segments complets trouvés"}
                return result , add_info
            else :
                return result

        N = len(segments) # nombres de segments

        # Calculer les probabilités théoriques
        t = max( max(segments) , self.max_lenght) # longueur maximale de segment à considérer

        probs = np.zeros(t-d + 1)


        for r in range(d, t):
            probs[r] = (factorial(d) / (d ** r)) * stirling(r - 1, d - 1)
        probs[t] = (1 - (factorial(d)/(d**(t-1)))*stirling(t-1,d) )

        # Compter les occurrences de chaque longueur
        length_counts = np.zeros(t + 1)
        for length in segments:
            if length <= t:
                length_counts[length] += 1

        # Appliquer le test du chi-carré
        expected_counts = N * probs
        chi2_stat = np.sum((length_counts - expected_counts) ** 2 / expected_counts)
        df = t - d + 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - chi2.cdf(chi2_stat, df)
        critical_value = chi2.ppf(1 - alpha, df)

        return chi2_from_other(observed=segments ,
                               expected=expected_counts,
                               info=info ,
                               alpha=alpha ,
                               extra_info= {
                                   "segments":segments,              "length_counts":length_counts,     "expected_counts":expected_counts,"d": d  })


class KSTest(Tests):
    """
    Test de Kolmogorov-Smirnov pour comparer la distribution empirique
    à une distribution uniforme.
    """


    def __str__(self) -> str:
        return "K-S Test"


    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict,Tuple[Dict]]:

        self._validate_input(data)

        # Vérifier que les données sont dans [0, 1]
        if np.min(data) < 0 or np.max(data) > 1:
            raise ValueError("Les données doivent être dans l'intervalle [0, 1]")

        # Trier les données
        sorted_data = np.sort(data)
        n = len(data)

        # Calculer la fonction de répartition empirique
        i = np.arange(1, n + 1)
        Fn = i / n

        # Calculer les différences maximales
        D_plus = np.max(Fn - sorted_data)
        D_minus = np.max(sorted_data - (i - 1) / n)
        D = max(D_plus, D_minus)

        # Calculer la statistique de test
        ks_stat = D * np.sqrt(n)

        # Calculer la p-value
        p_value = 1 - kstwobign.cdf(ks_stat)

        # Calculer la valeur critique
        critical_value = kstwobign.ppf(1 - alpha)

        result = {'stat_obs': ks_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value>alpha,
                  'violation': ks_stat > critical_value
                  }
        add_info={
                'ks_df':n ,
                "D_plus": D_plus,
                "D_minus": D_minus,
                "n": n
                }

        if info :
            return result , add_info
        else :
            return result


class MaximumTest(Tests):
    """
    Test des maximums pour détecter les tendances dans les séquences de nombres aléatoires.
    """

    def __init__(self, t: int = 5):
        """
            Initialise le test des maximums.

            Args:
                t: Nombre de groupes à considérer (défaut: 5)
        """
        if t < 2:
            raise ValueError("t doit être au moins 2")
        self.t = t

    def __str__(self) -> str:
        return f"Maximum_Test ( t={self.t} )"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:

        self._validate_input(data)

        n = len(data)
        if n < self.t:
            raise ValueError(f"La taille de l'échantillon ({n}) doit être au moins égale à t ({self.t})")

        # Diviser les données en t groupes
        group_size = n // self.t
        groups = [data[i:i + group_size] for i in range(0, n, group_size)]

        # Calculer les maximums de chaque groupe
        max_values = np.array([np.max(group) for group in groups])

        # Calculer la statistique de test
        # On utilise le test de tendance de Mann-Kendall
        stat, p_value = kendalltau(np.arange(self.t), max_values)

        # Calculer la valeur critique
        critical_value = norm.ppf(1 - alpha/2)

        result = {
                'stat_obs': stat,
                'stat_crit': critical_value,
                'p_value': p_value,
                'accept': p_value >=  alpha,
                'violation': stat > critical_value
                }

        add_info={
                'max_df':self.t-1,
                "max_values": max_values,
                "group_size": group_size,
                "t": self.t
            }


        if info :
            return result , add_info
        else :
            return result

