""" Module des Tests """
from math import factorial
from abc import ABC , abstractmethod
from collections import Counter
from itertools import permutations
from functools import cache
from typing import Literal , List,Dict,Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.stats import chi2 , kstwobign

# Les fonctions comunes à plusieurs tests
@cache
def stirling(n:int,k:int)->int:

    """
    Méthode pour calculer le nombre de stirling de deuxième espèce.
    """
    assert n >= 0 and k >= 0 , "n et k doivent être plus grand que 0."

    if n == k:
        return 1

    elif n == 0 or k == 0:
        return 0

    return k * stirling(n - 1, k) + stirling(n - 1, k - 1)

@cache
def input_type(data:NDArray)->Literal["integers","decimals"]:
    """
        Méthode pour déterminer le type de données
    """
    return "integers" if isinstance(data[0],int) else "decimals"

class Test(ABC):
    """
    Classe abstraite définissant un test d'hypothèse.
    """
    @abstractmethod
    def __str__(self)->str:
        """
        Méthode  permettant  de récupérer le nom du test.
        """

    @abstractmethod
    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:
        """
            teste  la séquence de nombres data

            Args:
                data : suite de nombre à tester
                alpha : seuil de signification (pourcentage d'erreur permit)
                info : si True retourne des infos supplementaires sur le test effectué

            Returns:
                _ : si info retourne un Dictionnaire contenant la stat seuil et observée Kᵣ , Kₐ , p-valeur ,
        """

class Chi2Test(Test):
    """
        classe qui impléménte le test de χ²

        Args:
            k:nombre de bins pour la catégorisation des nombres de data
    """
    def __init__(self, k:int=10):
        """Initialises le test du chi² avec k bins


        Args:
            k (int, optional): nombre de bins pour la catégorisation des nombres de data. Defaults to 10.
        """
        self.k = k

    def __str__(self):
        return "χ² Test "

    @staticmethod
    def group_low_frequencies(observed:NDArray, expected:NDArray, threshold=5)->Tuple[NDArray,NDArray]:
        """
            Regroupe les catégories adjacentes avec des fréquences attendues inférieures au seuil = threshold , s'il y'a des catégories faibles mais pas adjacentes  regroupe chacune d'elle avec sa voisine (directe) avec la plus petit effectif, et ce  jusqu'a ce que 80% des fréquences attendues soient supérieures au seuil
        """
        pass
        return observed, expected

    #group_low_frequencies n'est impléménté que dans cette classe car n'est utlisé que dans le cadre d'un test de chi2

    @staticmethod
    def chi_stat(observed:NDArray, expected:NDArray)->float:
        """
            Calcule la statistique du chi carré
        """

        observed, expected = Chi2Test.group_low_frequencies(observed, expected)
        return np.sum((observed - expected) ** 2 / expected) , observed.size - 1

    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:

        N = len(data)
        if input_type(data) == "integers":
            observed = np.bincount(data,minlength=np.max(data)+1)

        #les fréquences observées pour k bins
        observed , _ = np.histogram(data,bins=self.k)

        # les fréquences attendues pour k bins
        expected = np.full(self.k , N/self.k)


        #statistique du chi carré et degré de liberté
        chi2_stat , df = Chi2Test.chi_stat(observed, expected)

        #p-valeur
        p_value = float(chi2.cdf(chi2_stat, df))
        # Valeur/statistique critique
        Ka = float(chi2.ppf(1-alpha,df))

        return {
            "K": chi2_stat,
            "Ka": Ka,
            "p_value": p_value,
            "df": df
        }if info else p_value

class GapTest(Test):
    """
        classe qui représente le test de Gap

        Args:
            a : borne inférieur de l'intervalle pour le marquage
            b : borne supérieure de l'intervalle pour le marquage
            g : fonction de calcul du paramètre p de la loi gémetrique
    """
    def __init__(self, a=0.1,b=0.3):
        """Initialise le Gap Test avec une borne inférieur a et une borne supérieure b.

        Args:
            a (float, optional): Borne inférieure. Defaults to 0.1.
            b (float, optional): Borne supérieure. Defaults to 0.3.
        """
        self.a = a
        self.b = b
        self.p = b-a #probabilité d'être entre a et b sous l'hypothèse d'une loi uniforme

    def __str__(self):
        return "Gap Test "


    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:

        #selection des indices des nombres à marquer
        mark = (data >= self.a) & (data <= self.b)
        indices = np.where(mark)[0]

        if not indices.size: # il n'y a  pas de gaps
            return {"K":0,"Ka":0,"p_value":1,"df":0} if info else 1

        # Calculer les gaps entre les indices
        gaps = np.diff(indices) - 1
        max_gap = np.max(gaps)

        observed =  np.bincount(gaps , minlength=max_gap + 1)

        # calcul des fréquencesattend ues pour chaque gap

        expected = (gaps.size * self.p) * ( (1 - self.p) ** np.arange(max_gap + 1))

        # statistique du Chi carré , et déré de liberté
        chi2_stat  , df = Chi2Test.chi_stat(observed, expected)

        #p-valeur
        p_value = float(chi2.cdf(chi2_stat, df))

        # Valeur/statistique critique
        Ka = float(chi2.ppf(1-alpha,df))

        return {
            "K": chi2_stat,
            "Ka": Ka,
            "p_value": p_value,
            "df": df
        }if info else p_value

class PokerTest(Test):
    def __init__(self, n=5):  # nombre de cartes par joue
        self.n = n

    def __str__(self):
        return f"Test du Poker avec {self.n} cartes"

    @cache
    def test(self, data:NDArray, alpha:float=0.05, info:bool=False) -> float|Dict:
        # Vérifie si les données contiennent au moins self.n éléments uniques
        unique = np.unique(data)
        if len(unique) < 5:
            return {"K": 0, "Ka": 0, "p_value": 1.0, "df": 0} if info else 1.0

        # Générer toutes les combinaisons de 5 cartes
        from itertools import permutations
        all_permutations = List(permutations(unique, self.n))

        # Calculer la fréquence des runs
        run_counts = [0] * (len(all_permutations) + 1)
        for perm in all_permutations:
            try:
                # Trouver les positions consécutives
                indices = []
                current = perm[0]
                count = 1
                for i in range(1, len(perm)):
                    if data[i] == current and i - count == perm[i]:
                        indices.append(i)
                        count += 1
                    else:
                        break
                gap = max(indices) - min(indices) + 1

                # Si les cartes ne sont pas dans l'ordre croissant, ajuster la position
                if len(perm) != sorted(perm):
                    gap -= 1

                run_counts[gap] += 1
            except:
                pass

        observed = np.array(run_counts[1:])  # Omettre le zéro index
        total = data.size

        # Calcul des fréquences attendues
        expected = (total * self.n) / len(all_permutations)

        # Statistique du Chi carré et degré de liberté
        chi2_stat, df = Chi2Test.chi_stat(observed, expected)

        # Calcul p-valeur et statistique critique
        p_value = float(chi2.cdf(chi2_stat, df))
        Ka = float(chi2.ppf(1 - alpha, df))

        return {
            "K": chi2_stat,
            "Ka": Ka,
            "p_value": p_value,
            "df": df
        } if info else p_value

class CouponCollectorTest(Test):
    def __init__(self, probabilities=None, n_samples=1000):
        if probabilities is None:
            self.p = np.array([1/6] * 6)
        else:
            self.p = probabilities
        self.n_samples = n_samples

    def __str__(self):
        return f"Test du Collectionneur de {len(self.p)} coupons"

    @cache
    def test(self, data:NDArray, alpha:float=0.05, info:bool=False) -> float|Dict:
        # Calculer les effectifs observés pour chaque coupon
        observed = np.bincount(data, minlength=len(self.p))

        # Calculer les fréquences observées
        freq_observed = observed / data.size

        # Calculer les fréquences attendues (selon la Loi des nombres grands)
        freq_expected = self.p

        # Calculer les statistiques de Kolmogorov-Smirnov
        D_stat = np.max(np.abs(freq_observed - freq_expected))

        # Calculer la p-valeur et la statistique critique
        n = data.size
        D_statistic = D_stat * sqrt(n)
        p_value = 2 * (1 - stats.kstwobign.cdf(D_statistic))
        Ka = stats.kstwobign.ppf(1 - alpha)

        return {
            "D": D_statistic,
            "Ka": Ka,
            "p_value": p_value
        } if info else p_value

class PermutationTest(Test):
    def __init__(self, data1:NDArray, data2:NDArray, alpha=0.05):
        self.data1 = data1
        self.data2 = data2
        self.n_permutations = 1000

    def __str__(self):
        return "Test des Permutations"

    @cache
    def test(self, data:NDArray, alpha:float=0.05, info:bool=False) -> float|Dict:
        # Concaténer les données
        combined = np.concatenate([self.data1, self.data2])
        n = len(combined)

        # Générer toutes les permutations possibles
        from itertools import permutations
        perms = permutations(combined)

        # Calculer la statistiche test pour chaque permutation
        statistic_threshold = None
        for perm in perms:
            t_statistic = compute_test_statistic(perm, self.data1.size)
            if statistic_threshold is None or t_statistic > statistic_threshold:
                statistic_threshold = t_statistic

        # Calculer les statistiques pour les données réelles
        observed_statistic = compute_test_statistic(data, self.data1.size)

        # Comparer avec le seuil critique
        if observed_statistic >= statistic_threshold:
            return {"K": observed_statistic, "Ka": statistic_threshold, "p_value": 0.05} if info else True
        else:
            return {"K": observed_statistic, "Ka": statistic_threshold, "p_value": p_value} if info else False

def group_low_frequencies(data, k):
    # Split data into non-overlapping groups of size k
    n = len(data)
    if k > n:
        return []
    groups = [data[i:i+k] for i in range(0, n, k)]

    # Function to get order pattern ignoring duplicates
    def get_order_pattern(group):
        sorted_group = sorted(group)
        unique_sorted = List(sorted(set(sorted_group)))
        rank = {v: i+1 for i, v in enumerate(unique_sorted)}
        return tuple(rank[val] for val in group)

    # Get all possible permutation patterns (k!)
    all_patterns = set(permutations(range(1, k+1)))

    # Count observed frequencies
    pattern_counts = Counter()
    for g in groups:
        pattern = get_order_pattern(g)
        if len(pattern) == k:  # Ensure the group is of size k
            pattern_counts[pattern] += 1

    # Calculate expected counts under uniform distribution
    total_groups = len(groups)
    expected = {pattern: (total_groups / factorial(k)) for pattern in all_patterns}

    return pattern_counts, expected

class KSTest(Test):
    def __init__(self):
        pass

    def __str__(self):
        return "Test de Kolmogorov-Smirnov"

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> float | Dict:
        n = len(data)
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, n+1) / n

        theoretical_cdf = sorted_data

        D_plus = np.max(empirical_cdf - theoretical_cdf)
        D_minus = np.max(theoretical_cdf - (np.arange(n) / n))
        D_stat = max(D_plus, D_minus)

        p_value = 2 * (1 - kstwobign.cdf(D_stat * np.sqrt(n)))
        Ka = kstwobign.ppf(1 - alpha)

        return {
            "D": D_stat,
            "Ka": Ka,
            "p_value": p_value
        } if info else p_value

class MaxTest(Test):
    def __init__(self, t=3):
        self.t = t

    def __str__(self):
        return f"Test des Maximums avec sous-groupes de taille {self.t}"

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False) -> float | Dict:
        n = len(data) // self.t
        max_values = [max(data[i*self.t:(i+1)*self.t]) for i in range(n)]

        empirical_cdf = np.arange(1, n+1) / n
        theoretical_cdf = np.sort(max_values) ** self.t

        D_plus = np.max(empirical_cdf - theoretical_cdf)
        D_minus = np.max(theoretical_cdf - (np.arange(n) / n))
        D_stat = max(D_plus, D_minus)

        p_value = 2 * (1 - kstwobign.cdf(D_stat * np.sqrt(n)))
        Ka = kstwobign.ppf(1 - alpha)

        return {
            "D": D_stat,
            "Ka": Ka,
            "p_value": p_value
        } if info else p_value

# Example usage:
S = [2,5,5,6,9,5,6,9,8,4,2,8,7,0,7,3,0,9,4,0,2,4,0,3,2,1,0,6]
k = 3
pattern_counts, expecteds = group_low_frequencies(S, k)

# Print results
print("Observed pattern frequencies:", pattern_counts)
print("\nExpected pattern frequencies under uniform distribution:", expecteds)
