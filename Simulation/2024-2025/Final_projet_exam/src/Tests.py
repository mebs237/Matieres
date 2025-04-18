"""
    Module d'implementation des tests statistiques du χ² , Gap , poker , collectionneur de coupons , Maximum , Kolmogorov-smirnov

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Literal, Callable
from functools import cache
from math import factorial
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from collections import Counter
from itertools import permutations
import time
from Utils import *

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

        Args:
            data: Séquence de nombres à tester
            alpha: Niveau de signification (défaut: 0.05)
            info: Si True, retourne ( result , add_info )

        Returns
        ---
            dic : un dictionnaire :
            stat_obs (float) : statistique observée/calculée
            stat_crit (float) : valeur critique de la statistique
            p_value (float) : p valeur
            accept (bool) : on accepte l'hypothèse nulle ; le test n'a pas detecté une différence significative
            add_info (dict) : dictionnaire avec des information additionnelles sur le test
        """


    def _validate_input(self, data: NDArray) -> None:
        """Valide les données d'entrée."""
        if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype,np.number):
            raise TypeError("Les données doivent être un numpy.ndarray")
        if data.size == 0:
            raise ValueError("Les données ne peuvent pas être vides")
        if not np.all(np.isfinite(data)):
            raise ValueError("Les données doivent contenir uniquement des nombres finis")


class Chi2Test(Tests):
    """
        Test du Χ²
    """

    def __init__(self, k: int = 10, min_expected: float = 5.0, probabilities: Optional[NDArray] = None):
        """
        Initialise le test du Chi-carré.

        Args:
            k: Nombre de catégories (défaut: 10)
            min_expected: Fréquence minimale attendue pour au moins 80% des catégories (défaut: 5.0)
            probabilities: Probabilités théoriques pour chaque catégorie (défaut: None, distribution uniforme)
        """
        self.k = k
        self.min_expected = min_expected
        self.probabilities = probabilities

    def __str__(self) -> str:
        return f"Chi2_Test (k={self.k:.1f})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:

        self._validate_input(data)

        # Calculer les fréquences observées et attendues
        observed, expected = get_optimal_bins(data, k=self.k, probabilities=self.probabilities)

        # Regrouper les catégories si nécessaire
        observed, expected = group_low_frequence(observed, expected, seuil=self.min_expected)

        # Calculer la statistique du test
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        critical_value = stats.chi2.ppf(1 - alpha, df)

        result = {'stat_obs': chi2_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value >= alpha
                 }

        add_info = { "n_bins": df+1,"df" : df }

        if info :
            return result , add_info
        else :
            return result


class GapTest(Tests):
    """
    Test du Gap.
    """

    def __init__(self,
                 alpha_gap: float = 0.1,
                 beta_gap: float = 0.3,
                 max_gap: int = 30):
        """
        Initialise le test du Gap.

        Args:
            alpha_gap: Borne inférieure de l'intervalle (défaut: 0.1)
            beta_gap: Borne supérieure de l'intervalle (défaut: 0.3)
            max_gap: Longueur maximale de gap à considérer (défaut: 30)
        """
        if not 0 <= alpha_gap < beta_gap <= 1:
            raise ValueError("Les bornes doivent satisfaire 0 <= alpha_gap < beta_gap <= 1")

        self.alpha_gap = alpha_gap
        self.beta_gap = beta_gap
        self.max_gap = max_gap
        self.p = beta_gap - alpha_gap

    def __str__(self) -> str:
        return f"Gap Test ( [ {self.alpha_gap} , {self.beta_gap} ] )"

    @cache
    def _calculate_gap_probabilities(self, max_gap: int) -> NDArray:
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
                'accept' : False
            }

            add_info={"message": "Pas assez de marques pour calculer des gaps"}
            if info :
                return result , add_info
            else :
                return result

        # Calculer les gaps
        gaps = np.diff(indices) - 1
        max_observed_gap = np.max(gaps)
        # limite maximal de longueur de gap
        t = min(max_observed_gap, self.max_gap)

        # Compter les occurrences de chaque longueur de gap
        gap_counts = np.zeros(t + 1)
        for gap in gaps:
            if gap >= t:
                gap_counts[t] += 1
            else:
                gap_counts[gap] += 1

        # Calculer les probabilités théoriques
        expected_probs = self._calculate_gap_probabilities(t)
        expected_counts = len(gaps) * expected_probs

        # Appliquer le test du chi-carré
        chi2_stat = np.sum((gap_counts - expected_counts) ** 2 / expected_counts)
        df = t

        # Calculer la p-value et la valeur critique
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        critical_value = stats.chi2.ppf(1 - alpha, df)

        result = {
            'stat_obs':chi2_stat,
            'stat_crit':critical_value,
            'p_value':p_value,
            'accept':p_value >= alpha
                }

        add_info = { 'chi2_df':df , "max_gap": t}

        if info :
            return result , add_info
        else :
            return result


class PokerTest(Tests):
    """
    Test du Poker amélioré avec gestion optimisée des motifs.
    """

    def __init__(self, group_size: int = 5):
        """
        Initialise le test du Poker.

        Args:
            group_size: Taille des groupes à analyser (défaut: 5)
        """
        if group_size < 2:
            raise ValueError("La taille des groupes doit être au moins 2")
        self.group_size = group_size

    def __str__(self) -> str:
        return f"Poker_Test n = {self.group_size} "

    @cache
    def _get_pattern_type(self, group: NDArray) -> str:
        """
        Détermine le type de motif pour un groupe.
        """
        counts = Counter(group)
        values = list(counts.values())

        if len(counts) == self.group_size:
            return "all_different"
        elif len(counts) == self.group_size - 1:
            return "one_pair"
        elif len(counts) == self.group_size - 2:
            if 3 in values:
                return "three_of_a_kind"
            else:
                return "two_pairs"
        elif len(counts) == 2:
            if 4 in values:
                return "four_of_a_kind"
            else:
                return "full_house"
        else:
            return "five_of_a_kind"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict , Tuple[Dict]]:

        self._validate_input(data)

        # Vérifier qu'il y a assez de données
        if len(data) < self.group_size:
            raise ValueError(f"Pas assez de données pour former des groupes de taille {self.group_size}")

        # Diviser les données en groupes
        num_groups = len(data) // self.group_size
        groups = [data[i * self.group_size:(i + 1) * self.group_size]
                 for i in range(num_groups)]

        # Déterminer le type de motif pour chaque groupe
        pattern_types = [self._get_pattern_type(group) for group in groups]

        # Compter les occurrences de chaque motif
        pattern_counts = Counter(pattern_types)

        # Calculer les probabilités théoriques
        d = len(np.unique(data))  # Nombre de valeurs distinctes possibles
        if d < self.group_size:
            raise ValueError(f"Pas assez de valeurs distinctes ({d}) pour la taille de groupe ({self.group_size})")

        # Calculer les probabilités théoriques pour chaque motif
        theoretical_probs = {
            "all_different": (d * factorial(d-1)) / (d ** self.group_size),
            "one_pair": (self.group_size * (self.group_size-1) * d * factorial(d-2)) / (2 * d ** self.group_size),
            "two_pairs": (self.group_size * (self.group_size-1) * (self.group_size-2) * (self.group_size-3) * d * factorial(d-3)) / (4 * d ** self.group_size),
            "three_of_a_kind": (self.group_size * (self.group_size-1) * (self.group_size-2) * d * factorial(d-2)) / (6 * d ** self.group_size),
            "full_house": (self.group_size * (self.group_size-1) * (self.group_size-2) * d * (d-1)) / (6 * d ** self.group_size),
            "four_of_a_kind": (self.group_size * (self.group_size-1) * (self.group_size-2) * (self.group_size-3) * d) / (24 * d ** self.group_size),
            "five_of_a_kind": 1 / (d ** (self.group_size-1))
        }

        # Calculer les fréquences attendues
        expected_counts = {pattern: num_groups * prob
                         for pattern, prob in theoretical_probs.items()}

        # Appliquer le test du chi-carré
        chi2_stat = sum((pattern_counts.get(pattern, 0) - expected_counts[pattern]) ** 2
                       / expected_counts[pattern]
                       for pattern in theoretical_probs.keys())
        df = len(theoretical_probs) - 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        critical_value = stats.chi2.ppf(1 - alpha, df)

        result = {'stat_obs': chi2_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value >= alpha
                  }

        add_info={
                'df':df ,
                "pattern_counts": dict(pattern_counts),
                "expected_counts": expected_counts,
                "theoretical_probs": theoretical_probs
                }


        if info :
            return result , add_info
        else :
            return result


class CouponCollectorTest(Tests):
    """
    Test du Collectionneur de Coupons amélioré.
    """

    def __init__(self, d: Optional[int] = None):
        """
        Initialise le test du Collectionneur de Coupons.

        Args:
            d: Nombre de valeurs distinctes possibles (si None, utilise le nombre de valeurs uniques)
        """
        self.d = d

    def __str__(self) -> str:
        return f"Coupon_Collector_Test (d={self.d})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test du Collectionneur de Coupons.
        """
        self._validate_input(data)

        # Vérifier que les données sont des entiers
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
                  'accept': True
                  }

            add_info={"message": "Pas de segments complets trouvés"}

            if info :
                return result , add_info
            else :
                return result

        # Calculer les probabilités théoriques
        max_length = max(segments)
        probs = np.zeros(max_length + 1)

        for r in range(d, max_length + 1):
            probs[r] = (factorial(d) / (d ** r)) * stirling(r - 1, d - 1)

        # Compter les occurrences de chaque longueur
        length_counts = np.zeros(max_length + 1)
        for length in segments:
            if length <= max_length:
                length_counts[length] += 1

        # Appliquer le test du chi-carré
        expected_counts = len(segments) * probs
        chi2_stat = np.sum((length_counts - expected_counts) ** 2 / expected_counts)
        df = max_length - d + 1

        # Calculer la p-value et la valeur critique
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        critical_value = stats.chi2.ppf(1 - alpha, df)

        result = {'stat_obs': chi2_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value>=alpha
                  }

        add_info={
                "chi2_df":df,
                "segments": segments,
                "length_counts": length_counts,
                "expected_counts": expected_counts,
                "d": d
            }

        if info :
            return result , add_info
        else :
            return result


class KSTest(Tests):
    """
    Test de Kolmogorov-Smirnov pour comparer la distribution empirique
    à une distribution uniforme.
    """

    def __init__(self):
        """Initialise le test de Kolmogorov-Smirnov."""


    def __str__(self) -> str:
        return "Kolmogorov-Smirnov_Test"


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
        p_value = 1 - stats.kstwobign.cdf(ks_stat)

        # Calculer la valeur critique
        critical_value = stats.kstwobign.ppf(1 - alpha)

        result = {'stat_obs': ks_stat,
                  'stat_crit': critical_value,
                  'p_value': p_value,
                  'accept': p_value>alpha
                  }
        add_info={
                'ks_df':n ,
                "D_plus": D_plus,
                "D_minus": D_minus,
                "D": D,
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
        stat, p_value = stats.kendalltau(np.arange(self.t), max_values)

        # Calculer la valeur critique
        critical_value = stats.norm.ppf(1 - alpha/2)

        result = {
                'stat_obs': stat,
                'stat_crit': critical_value,
                'p_value': p_value,
                'accept': p_value >=  alpha
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

