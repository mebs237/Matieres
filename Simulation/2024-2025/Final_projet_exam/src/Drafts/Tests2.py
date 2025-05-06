from abc import ABC, abstractmethod
from math import factorial, sqrt
import numpy as np
from scipy.stats import chi2, kstwobign
from collections import Counter
from itertools import permutations
from functools import cache
from typing import Dict, Tuple, List, Literal, Union, Optional
from numpy.typing import NDArray

@cache
def stirling(n: int, k: int) -> int:
    """
    Calcule le nombre de Stirling de deuxième espèce.
    """
    if n < 0 or k < 0:
        raise ValueError("n et k doivent être des entiers positifs")
    
    if n == k:
        return 1
    elif n == 0 or k == 0:
        return 0
    
    return k * stirling(n - 1, k) + stirling(n - 1, k - 1)

def input_type(data: NDArray) -> Literal["integers", "decimals"]:
    """
    Détermine le type de données (entiers ou décimaux).
    """
    return "integers" if np.issubdtype(data.dtype, np.integer) else "decimals"

class Test(ABC):
    """
    Classe abstraite définissant un test d'hypothèse.
    """
    @abstractmethod
    def __str__(self) -> str:
        """
        Retourne le nom du test.
        """
        pass

    @abstractmethod
    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        """
        Teste la séquence de nombres data.

        Args:
            data: Suite de nombres à tester
            alpha: Seuil de signification (pourcentage d'erreur permis)
            info: Si True, retourne des informations supplémentaires

        Returns:
            Si info est True, retourne un dictionnaire contenant les statistiques du test.
            Sinon, retourne la p-valeur.
        """
        pass

class Chi2Test(Test):
    """
    Implémente le test du chi-carré.
    """
    def __init__(self, k: int = 10):
        """
        Initialise le test du chi-carré avec k bins.

        Args:
            k: Nombre de bins pour la catégorisation des données
        """
        self.k = k

    def __str__(self) -> str:
        return f"χ² Test (k={self.k})"

    @staticmethod
    def group_low_frequencies(observed: NDArray, expected: NDArray, threshold: float = 5) -> Tuple[NDArray, NDArray]:
        """
        Regroupe les catégories adjacentes avec des fréquences attendues inférieures au seuil.
        
        Args:
            observed: Fréquences observées
            expected: Fréquences attendues
            threshold: Seuil minimal pour les fréquences attendues
            
        Returns:
            Tuple contenant les fréquences observées et attendues après regroupement
        """
        # Copie des tableaux pour éviter de modifier les originaux
        obs = observed.copy()
        exp = expected.copy()
        
        # Tant qu'il y a des catégories faibles adjacentes
        i = 0
        while i < len(exp) - 1:
            if exp[i] < threshold or exp[i+1] < threshold:
                # Fusionner les catégories i et i+1
                obs[i] += obs[i+1]
                exp[i] += exp[i+1]
                # Supprimer la catégorie i+1
                obs = np.delete(obs, i+1)
                exp = np.delete(exp, i+1)
            else:
                i += 1
                
        return obs, exp

    @staticmethod
    def chi_stat(observed: NDArray, expected: NDArray) -> Tuple[float, int]:
        """
        Calcule la statistique du chi-carré et le degré de liberté.
        
        Args:
            observed: Fréquences observées
            expected: Fréquences attendues
            
        Returns:
            Tuple contenant la statistique du chi-carré et le degré de liberté
        """
        observed, expected = Chi2Test.group_low_frequencies(observed, expected)
        return np.sum(((observed - expected) ** 2) / expected), observed.size - 1

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        N = len(data)
        
        # Stratégie différente selon le type de données
        if input_type(data) == "integers":
            # Pour les entiers, utiliser np.bincount pour compter les occurrences
            max_value = np.max(data)
            observed = np.bincount(data, minlength=max_value + 1)
            # Ajuster k si nécessaire
            k = min(self.k, len(observed))
            # Regrouper les données en k catégories si nécessaire
            if len(observed) > k:
                # Répartir les données en k catégories
                new_observed = np.zeros(k)
                for i in range(len(observed)):
                    bin_idx = min(int(i * k / len(observed)), k - 1)
                    new_observed[bin_idx] += observed[i]
                observed = new_observed
        else:
            # Pour les décimaux, utiliser np.histogram
            observed, _ = np.histogram(data, bins=self.k, range=(0, 1))
            k = self.k
            
        # Fréquences attendues pour k bins
        expected = np.full(k, N / k)
        
        # Statistique du chi-carré et degré de liberté
        chi2_stat, df = Chi2Test.chi_stat(observed, expected)
        
        # p-valeur (1 - CDF car nous voulons P(X > chi2_stat))
        p_value = 1 - chi2.cdf(chi2_stat, df)
        
        # Valeur critique
        critical_value = chi2.ppf(1 - alpha, df)
        
        if info:
            return {
                "statistic": chi2_stat,
                "critical_value": critical_value,
                "p_value": p_value,
                "df": df,
                "reject_null": p_value < alpha
            }
        else:
            return p_value

class GapTest(Test):
    """
    Implémente le test du Gap.
    """
    def __init__(self, a: float = 0.1, b: float = 0.3):
        """
        Initialise le test du Gap avec les bornes de l'intervalle.
        
        Args:
            a: Borne inférieure de l'intervalle
            b: Borne supérieure de l'intervalle
        """
        if not 0 <= a < b <= 1:
            raise ValueError("Les bornes doivent satisfaire 0 <= a < b <= 1")
        
        self.a = a
        self.b = b
        self.p = b - a  # Probabilité d'être dans l'intervalle [a,b] sous H0

    def __str__(self) -> str:
        return f"Gap Test ([{self.a}, {self.b}])"

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        # Vérifier que les données sont dans [0,1]
        if np.min(data) < 0 or np.max(data) > 1:
            raise ValueError("Les données doivent être dans l'intervalle [0,1]")
        
        # Marquer les nombres dans l'intervalle [a,b]
        marks = (data >= self.a) & (data <= self.b)
        indices = np.where(marks)[0]
        
        if len(indices) <= 1:
            # Pas assez de marques pour calculer des gaps
            return 1.0 if not info else {"statistic": 0, "critical_value": 0, "p_value": 1.0, "df": 0, "reject_null": False}
        
        # Calculer les gaps (distances entre les indices consécutifs - 1)
        gaps = np.diff(indices) - 1
        
        # Histogramme des gaps
        max_gap = np.max(gaps)
        t = min(max_gap, 30)  # Limiter t pour éviter des catégories avec trop peu d'occurrences
        
        # Compter les occurrences de chaque longueur de gap
        gap_counts = np.zeros(t + 1)
        for gap in gaps:
            if gap <= t:
                gap_counts[gap] += 1
            else:
                gap_counts[t] += 1
        
        # Probabilités théoriques pour chaque longueur de gap
        p = self.p
        q = 1 - p
        
        # Probabilités théoriques pour chaque longueur de gap
        expected_probs = np.zeros(t + 1)
        for i in range(t):
            expected_probs[i] = p * (q ** i)
        expected_probs[t] = q ** t  # Probabilité pour les gaps >= t
        
        # Fréquences attendues
        expected_counts = len(gaps) * expected_probs
        
        # Statistique du chi-carré et degré de liberté
        chi2_stat, df = Chi2Test.chi_stat(gap_counts, expected_counts)
        
        # p-valeur
        p_value = 1 - chi2.cdf(chi2_stat, df)
        
        # Valeur critique
        critical_value = chi2.ppf(1 - alpha, df)
        
        if info:
            return {
                "statistic": chi2_stat,
                "critical_value": critical_value,
                "p_value": p_value,
                "df": df,
                "reject_null": p_value < alpha
            }
        else:
            return p_value

class PokerTest(Test):
    """
    Implémente le test du Poker.
    """
    def __init__(self, group_size: int = 5, k: int = 10):
        """
        Initialise le test du Poker.
        
        Args:
            group_size: Taille des groupes à analyser
            k: Nombre de catégories pour les nombres décimaux
        """
        if group_size < 2:
            raise ValueError("La taille des groupes doit être au moins 2")
        
        self.group_size = group_size
        self.k = k

    def __str__(self) -> str:
        return f"Poker Test (n={self.group_size})"

    def _get_pattern_type(self, group: NDArray) -> str:
        """
        Détermine le type de motif pour un groupe.
        
        Args:
            group: Groupe de nombres
            
        Returns:
            Type de motif (ex: "all_different", "one_pair", etc.)
        """
        # Compter les occurrences de chaque valeur
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

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        n = len(data)
        
        # Vérifier qu'il y a assez de données
        if n < self.group_size:
            raise ValueError(f"Pas assez de données pour former des groupes de taille {self.group_size}")
        
        # Pour les données décimales, discrétiser en k catégories
        if input_type(data) == "decimals":
            data = np.floor(data * self.k).astype(int)
        
        # Diviser les données en groupes
        num_groups = n // self.group_size
        groups = [data[i * self.group_size:(i + 1) * self.group_size] for i in range(num_groups)]
        
        # Déterminer le type de motif pour chaque groupe
        pattern_types = [self._get_pattern_type(group) for group in groups]