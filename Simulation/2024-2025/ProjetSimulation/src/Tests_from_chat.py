"""
Module d'implémentation des tests statistiques.

Tests_classes
--------------
Les classes sont :

- ``Tests`` : classe abstraite servant d'interface aux autres classes, les
  obligeant à implémenter la méthode ``test``.
- ``Chi2Test`` : test du **χ²** qui analyse la fréquence d'apparition de
  catégories dans une séquence.
- ``GapTestv2`` : test du **Gap** (version simplifiée).
- ``PokerTest`` : test du **Poker** généralisé (motifs exacts).
- ``PokerTestv2`` : version simplifiée du test du Poker (nombre de valeurs distinctes).
- ``CouponCollectorTest`` : test du **Collectionneur de Coupons**.
- ``MaximumTest`` : test des **Maximum**.
- ``KSTest`` : test de **Kolmogorov-Smirnov**.

Fontions auxilliaires
---------------------
Ces fonctions sont utilisées dans l'implémentation des tests :

- `discretize` : discrétiser une séquence de nombres réels en entiers (classes).
- `stirling` : calcul du nombre de Stirling de 2e espèce.
- `name_from_config`, `invert_dict` : fonctions utilitaires pour le test du Poker.

Utilisation
-----------
Pour une `sequence`, on teste l'hypothèse **H₀ : la séquence suit une
distribution uniforme sur [0,1]** avec le test statistique T :

1. Instanciation : créer une instance `t` de la classe du test voulu.
2. Application : exécuter la méthode `test(sequence, alpha)` sur `t`.
3. Interprétation : ``result["accept"]`` indique si on *accepte* l'hypothèse nulle
   (True : on ne détecte pas de différence significative).

Exemple
-------
>>> sequence = [5,9,3,1,4,9,6,2,3,4,...]
>>> test = Chi2Test()
>>> res_test = test.test(sequence, alpha=0.01)
>>> print(f"Selon le test du χ², la séquence est uniforme ? {res_test['accept']} ")
"""
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Union, Optional, Tuple, Callable
from functools import lru_cache
from math import factorial, comb
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, kstwobign

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
            ) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test statistique sur les données.

        Args:
            data : séquence de nombres (numpy.ndarray) à tester.
            alpha : niveau de signification (défaut 0.05).
            info : si True, retourne (result, add_info).

        Returns:
            Dictionnaire avec : stat_obs, stat_crit, p_value, accept.
            Si info=True, retourne (result, add_info) avec détails supplémentaires.
        """

    def _validate_input(self, data: NDArray) -> None:
        """Vérifie que `data` est un array numpy non vide de nombres finis."""
        if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype, np.number):
            raise TypeError("Les données doivent être un numpy.ndarray de nombres")
        if data.size == 0:
            raise ValueError("Les données ne peuvent pas être vides")
        if not np.all(np.isfinite(data)):
            raise ValueError("Les données doivent contenir uniquement des nombres finis")

    @staticmethod
    @lru_cache(maxsize=None)
    def stirling(n: int, k: int) -> int:
        """
        Calcule le nombre de Stirling de deuxième espèce S(n, k).
        Nombre de façons de partitionner un ensemble de taille n en k sous-ensembles.
        """
        if n < 0 or k < 0:
            raise ValueError("n et k doivent être des entiers positifs")
        if n == k:
            return 1
        if n == 0 or k == 0:
            return 0
        return k * Tests.stirling(n - 1, k) + Tests.stirling(n - 1, k - 1)

    @staticmethod
    def discretize(data: NDArray,
                   a: Union[int, float] = 0,
                   b: Union[int, float] = 1,
                   n_bins: Optional[int] = None,
                   binning: str = "quantile") -> Tuple[NDArray, int]:
        """
        Discrétisation d'une séquence de nombres.

        Si `data` est déjà entière, renvoie (data, nombre_valeurs_uniques).
        Sinon, subdivise [min(data), max(data)] en n_bins intervalles.
        """
        arr = np.array(data)
        if np.issubdtype(arr.dtype, np.integer):
            return arr, len(np.unique(arr))

        d = n_bins if n_bins is not None else int(1 + np.log2(len(arr)))
        if binning == "quantile":
            bins = np.quantile(arr, np.linspace(a, b, d + 1)[1:-1])
        elif binning == "uniform":
            bins = np.linspace(np.min(arr), np.max(arr), d + 1)[1:-1]
        else:
            raise NotImplementedError("Binning non supportée (quantile ou uniform).")

        discretized = np.digitize(arr, bins)
        return discretized, d

class Chi2Test(Tests):
    """
    Test du χ² pour un ajustement de distribution uniforme.

    Parameters:
        k : nombre de catégories (défaut 10).
        min_expected : effectif minimal attendu (défaut 5.0).
        probabilities : probabilités théoriques pour chaque catégorie (uniforme si None).
    """
    def __init__(self,
                 k: int = 10,
                 min_expected: float = 5.0,
                 probabilities: Optional[NDArray] = None) -> None:
        self.k = k
        self.min_expected = min_expected
        self.probabilities = probabilities

    def __str__(self) -> str:
        return f"Chi2_Test(k={self.k})"

    @staticmethod
    def group_low_frequence(observed: NDArray,
                            expected: NDArray,
                            min_expected: float = 5.0) -> Tuple[NDArray, NDArray]:
        """
        Regroupe les classes à effectifs attendus < min_expected en fusionnant
        avec des classes voisines, tant que possible.
        """
        observed = np.array(observed, copy=True)
        expected = np.array(expected, copy=True)
        if len(observed) != len(expected):
            raise ValueError("Taille differentes pour observed et expected.")

        obs_new = observed.copy()
        exp_new = expected.copy()
        # Boucle de regroupement
        while len(obs_new) > 1 and np.any(exp_new < min_expected):
            idx = np.argmin(exp_new)
            if idx == 0:
                adj = 1
            elif idx == len(exp_new) - 1:
                adj = idx - 1
            else:
                adj = idx - 1 if exp_new[idx - 1] <= exp_new[idx + 1] else idx + 1
            i_min, i_adj = sorted((idx, adj))
            obs_new[i_min] += obs_new[i_adj]
            exp_new[i_min] += exp_new[i_adj]
            obs_new = np.delete(obs_new, i_adj)
            exp_new = np.delete(exp_new, i_adj)
        return obs_new, exp_new

    @staticmethod
    def get_optimal_bins(data: NDArray,
                         k: int = 10,
                         probabilities: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
        """
        Calcule bins optimaux (Sturges) ou selon unique values si données entières.

        Args:
            data: données en entrée (np.ndarray).
            k: nombre de classes souhaité.
            probabilities: éventuelles probabilités théoriques.
        Returns:
            (observed, expected) effectifs observés et attendus par classe.
        """
        arr = np.array(data)
        n = len(arr)
        if n == 0:
            raise ValueError("Les données ne peuvent pas être vides")

        # Cas des données entières: utiliser les valeurs uniques
        if np.issubdtype(arr.dtype, np.integer):
            uniq = np.unique(arr)
            d = len(uniq)
            if d <= k:
                observed = np.bincount(arr, minlength=arr.max()+1)
                observed = observed[uniq]  # effectifs des valeurs présentes
                if probabilities is None:
                    expected = np.full(len(uniq), n / d)
                else:
                    probs = np.array(probabilities)
                    if len(probs) != len(uniq):
                        raise ValueError("Probabilités vs valeurs uniques mismatch")
                    expected = n * probs
                return observed, expected

        # Sinon, histogramme classique
        k_eff = min(k, int(1 + np.log2(n)))
        observed, _ = np.histogram(arr, bins=k_eff)
        if probabilities is None:
            expected = np.full(k_eff, n / k_eff)
        else:
            probs = np.array(probabilities)
            if len(probs) != k_eff:
                raise ValueError("len(probabilities) != nombre de classes bins")
            expected = n * probs
        return observed, expected

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        self._validate_input(data)
        observed, expected = self.get_optimal_bins(data, k=self.k, probabilities=self.probabilities)
        observed, expected = self.group_low_frequence(observed, expected, min_expected=self.min_expected)
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1
        p_value = 1 - chi2.cdf(chi2_stat, df)
        crit = chi2.ppf(1 - alpha, df)
        result = {'stat_obs': chi2_stat,
                  'stat_crit': crit,
                  'p_value': p_value,
                  'accept': p_value > alpha}
        if info:
            return result, {'df': df, 'observed': observed, 'expected': expected}
        return result

def chi2_from_other(observed: NDArray,
                    expected: Optional[NDArray] = None,
                    probabilities: Optional[NDArray] = None,
                    alpha: float = 0.05,
                    min_expected: float = 5.0,
                    info: bool = False,
                    extra_info: Optional[Dict] = None) -> Union[Dict, Tuple[Dict]]:
    """
    Test du χ² à partir d'effectifs observés (et attendus ou probabilités données).
    """
    obs = np.array(observed)
    n = np.sum(obs)
    if expected is None:
        if probabilities is None:
            exp = np.full_like(obs, n / len(obs))
        else:
            probs = np.array(probabilities)
            if len(probs) != len(obs):
                raise ValueError("Probabilités et observed doivent avoir même longueur")
            exp = n * probs
    else:
        exp = np.array(expected)
        if len(exp) != len(obs):
            raise ValueError("Observed/expected de longueurs différentes")
    obs_grp, exp_grp = Chi2Test.group_low_frequence(obs, exp, min_expected)
    chi2_stat = np.sum((obs_grp - exp_grp) ** 2 / exp_grp)
    df = len(obs_grp) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    crit = chi2.ppf(1 - alpha, df)
    result = {"stat_obs": chi2_stat,
              "stat_crit": crit,
              "p_value": p_value,
              "accept": p_value > alpha}
    if info:
        add = {"df": df, "observed": obs_grp, "expected": exp_grp}
        if extra_info:
            add.update(extra_info)
        return result, add
    return result

def name_from_config(config) -> str:
    """
    Donne un nom textuel à une configuration {taille: nombre} de motifs.
    Exemple : {3:2, 2:1} -> "2–3 uplet, 1–2 uplet".
    """
    parts = [f"{count}–{k}–uplet" for k, count in sorted(config.items(), reverse=True)]
    return ", ".join(parts) if parts else "all different"

def invert_dict(motifs: Dict[str, Dict[int, int]]) -> Dict[Tuple[Tuple[int, int], ...], str]:
    """
    Inverse un dict de motifs (nom->config) en config -> nom.
    """
    inv = {}
    for name, cfg in motifs.items():
        key = tuple(sorted(cfg.items()))
        inv[key] = name
    return inv

class GapTestv2(Tests):
    """
    Test du Gap (version simplifiée).
    On marque les éléments dans [alpha_gap, beta_gap], on étudie la
    distribution des espacements (gaps) entre marques.
    """
    def __init__(self,
                 alpha_gap: float = 0.1,
                 beta_gap: float = 0.4,
                 max_gap: int = 30) -> None:
        if not 0 <= alpha_gap < beta_gap <= 1:
            raise ValueError("0 <= alpha_gap < beta_gap <= 1 requis")
        self.alpha_gap = alpha_gap
        self.beta_gap = beta_gap
        self.max_gap = max_gap
        self.p = beta_gap - alpha_gap

    def __str__(self) -> str:
        return f"Gap_Test([{self.alpha_gap}, {self.beta_gap}])"

    def _compute_gap_probabilities(self, max_gap: int) -> NDArray:
        """
        Calcule les probabilités P(Gap = k) pour k=0..max_gap.
        """
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
        Exécute le test du Gap sur `data`.
        """
        self._validate_input(data)
        arr = np.array(data)
        max_data , min_data = np.max(arr) , np.min(arr)
        if min_data < 0 or max_data > 1:
            arr = (arr - min_data) / (max_data-min_data)

        marks = (arr >= self.alpha_gap) & (arr <= self.beta_gap)
        indices = np.where(marks)[0]
        if len(indices) <= 1:
            result = {'stat_obs': 0, 'stat_crit': 0, 'p_value': 1.0, 'accept': False}
            if info:
                return result, {"message": "Pas assez de marques pour gap"}
            return result

        gaps = np.diff(indices) - 1
        max_observed_gap = np.max(gaps)
        t = min(max_observed_gap, self.max_gap)
        gaps = np.clip(gaps, 0, t)

        expected_probs = self._compute_gap_probabilities(t)
        chi2_test = Chi2Test(k=t+1, probabilities=expected_probs)
        result = chi2_test.test(data=gaps, alpha=alpha, info=info)
        if info and isinstance(result, tuple):
            result[1].update({"max_gap": t})
        return result

class PokerTest(Tests):
    """
    Test du Poker généralisé (t>=2), catégorise les motifs exacts.
    """
    def __init__(self, group_size: int = 5) -> None:
        """
        Initialisation du test du Poker.
        """
        if group_size < 2:
            raise ValueError("Taille des groupes >= 2 requise")
        self.group_size = group_size
        self._motifs = PokerTest.generate_motifs(group_size)

    def __str__(self) -> str:
        return f"Poker_Test(t={self.group_size})"

    @staticmethod
    def partitions(n: int, max_part: Optional[int] = None):
        """
        Génère toutes les partitions de n en entiers (listes décroissantes).
        """
        if n == 0:
            yield []
        else:
            if max_part is None or max_part > n:
                max_part = n
            for k in range(max_part, 0, -1):
                for rest in PokerTest.partitions(n - k, k):
                    yield [k] + rest

    @staticmethod
    def generate_motifs(t: int) -> Dict[str, Dict[int, int]]:
        """
        Construit un dict de motifs pour groupes de taille t.
        Renvoie {nom_motif: {taille: count}}.
        """
        mapping: Dict[str, Dict[int, int]] = {}
        # Noms prédéfinis pour t<=5
        base_names = {
            2: {(2,): "paire", (1,1): "all different"},
            3: {(3,): "brelan", (2,1): "paire", (1,1,1): "all different"},
            4: {(4,): "carré", (3,1): "brelan",
                (2,2): "double paire", (2,1,1): "paire",
                (1,1,1,1): "all different"},
            5: {(5,): "quintuplet", (4,1): "carré",
                (3,2): "full", (3,1,1): "brelan",
                (2,2,1): "double paire", (2,1,1,1): "paire",
                (1,1,1,1,1): "all different"}
        }
        for p in PokerTest.partitions(t):
            cfg = Counter(p)
            freq = tuple(sorted(p, reverse=True))
            if freq == (t,):
                name = "all identical"
            elif 2 <= t <= 5:
                name = base_names.get(t, {}).get(freq, name_from_config(cfg))
            else:
                name = name_from_config(dict(cfg))
            if name in mapping and mapping[name] != dict(cfg):
                raise ValueError(f"Nom dupliqué pour configs différentes: {name}")
            mapping[name] = dict(cfg)
        return mapping

    @staticmethod
    def assign_motifs(group: NDArray, motifs: Dict[str, Dict[int, int]]) -> str:
        """
        Assigne à `group` (taille t) le motif correspondant parmi `motifs`.
        """
        freq = Counter(group)
        cfg = Counter(freq.values())
        key = tuple(sorted(cfg.items()))
        return invert_dict(motifs).get(key, "inconnu")

    @staticmethod
    def _probabilities(t: int, d: int, motifs: Dict[str, Dict[int, int]]) -> Dict[str, float]:
        """
        Calcule la probabilité théorique de chaque motif pour groupes de taille t,
        d valeurs possibles.
        """
        probs: Dict[str, float] = {}
        for name, cfg in motifs.items():
            r = sum(cfg.values())  # nombre de valeurs distinctes dans ce motif
            probs[name] = Tests.stirling(t, r) * comb(d, r) * factorial(r) / (d ** t)
        return probs

    def test(self, data: NDArray, alpha: float = 0.05, info: bool = False):
        self._validate_input(data)
        n = len(data)
        # Discrétisation si flottants
        if not np.issubdtype(data.dtype, np.integer):
            d_bins = min(self.group_size, int(1 + np.log10(n)))
            data_new, d = self.discretize(data, n_bins=d_bins)
        else:
            data_new = data
            d = len(np.unique(data))

        n_group = n // self.group_size
        if n_group < 1:
            raise ValueError(f"Pas assez de données pour groupes de taille {self.group_size}")
        groups = data_new[:n_group * self.group_size].reshape(n_group, self.group_size)

        # Comptage observé des motifs
        obs_counts: Dict[str, int] = {}
        for grp in groups:
            name = self.assign_motifs(grp, self._motifs)
            obs_counts[name] = obs_counts.get(name, 0) + 1

        probs = self._probabilities(self.group_size, d, self._motifs)
        observed = np.array([obs_counts.get(name, 0) for name in self._motifs.keys()])
        expected_probs = np.array([probs.get(name, 0) for name in self._motifs.keys()])

        chi2_test = Chi2Test(k=len(self._motifs), probabilities=expected_probs)
        result = chi2_test.test(observed, alpha=alpha, info=info)
        if info and isinstance(result, tuple):
            result[1].update({
                "motifs": self._motifs,
                "observed_counts": obs_counts,
                "theoretical_probs": probs
            })
        return result

class PokerTestv2(Tests):
    """
    Version simplifiée du test du Poker : on compte le nombre de valeurs distinctes
    dans chaque groupe de taille t.
    """
    def __init__(self, group_size: int = 5, n_bins: Optional[int] = None) -> None:
        """
        Initialisation du test du Poker (version v2).
        """
        if group_size < 2:
            raise ValueError("Taille des groupes >= 2 requise")
        self.group_size = group_size
        self.n_bins = n_bins

    def __str__(self) -> str:
        return f"Poker_Test_V2(t={self.group_size})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False):
        """
        Exécute le test du Poker (v2) sur `data`.
        Catégories : nombre de valeurs distinctes dans chaque groupe.
        """
        self._validate_input(data)
        n = len(data)
        data_new, d = self.discretize(data, n_bins=self.n_bins)
        n_group = n // self.group_size
        if n_group < 1:
            raise ValueError(f"Pas assez de données pour groupes de taille {self.group_size}")
        groups = data_new[:n_group * self.group_size].reshape(n_group, self.group_size)

        distinct_values = [len(Counter(group)) for group in groups]
        d = min(self.group_size, d)  # nombre max de valeurs distinctes possible

        probs = np.zeros(d)
        for r in range(1, d):
            probs[r] = Tests.stirling(self.group_size, r) * comb(d, r) * factorial(r) / (d ** self.group_size)

        chi2_test = Chi2Test(k=d, probabilities=probs)
        result = chi2_test.test(np.array(distinct_values), alpha=alpha, info=info)
        if info and isinstance(result, tuple):
            result[1].update({
                "group_size": self.group_size,
                "n_bins": self.n_bins,
                "theoretical_probs": probs
            })
        return result

class CouponCollectorTest(Tests):
    """
    Test du Collectionneur de Coupons.
    """
    def __init__(self, d: Optional[int] = None, max_length: int = 30):
        """
        Initialise le test du Collectionneur de Coupons.

        d : nombre de coupons distincts (None = utiliser unique(data)).
        max_length : longueur maximale de segment considérée.
        """
        self.d = d
        self.max_length = max_length

    def __str__(self) -> str:
        return f"Coupon_Collector_Test(d={self.d})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        self._validate_input(data)
        n = len(data)
        # Discrétiser pour travailler en entiers
        data_new, d = self.discretize(data, n_bins=self.d)
        # Calculer les longueurs de segments complétant un cycle
        len_segments = []
        current_set = set()
        seg_len = 0
        for val in data_new:
            current_set.add(val)
            seg_len += 1
            if len(current_set) == d:
                len_segments.append(seg_len)
                current_set.clear()
                seg_len = 0
        N = len(len_segments)
        if N == 0:
            result = {'stat_obs': 0, 'stat_crit': 0, 'p_value': 1.0, 'accept': False}
            if info:
                return result, {"message": "Pas de segments complets trouvés"}
            return result
        t = min(max(len_segments), self.max_length, n)
        probs = np.zeros(t + 1)
        # P(length = r) pour r = d..t-1
        for r in range(d, t):
            probs[r] = (factorial(d) / (d ** r)) * Tests.stirling(r - 1, d - 1)
        probs[t] = 1 - np.sum(probs[:t])
        observed = np.zeros(t + 1)
        for length in len_segments:
            if length <= t:
                observed[length] += 1
        chi2_test = Chi2Test(k=t+1, probabilities=probs)
        result = chi2_test.test(observed, alpha=alpha, info=info)
        if info and isinstance(result, tuple):
            result[1].update({
                "max_length": t,
                "length_counts": observed,
                "expected_probs": probs
            })
        return result

class KSTest(Tests):
    """
    Test de Kolmogorov-Smirnov (empirique vs distribution donnée).
    """
    def __init__(self,
                 cdf: Optional[Callable[[NDArray], NDArray]] = None):
        """
        Initialise le test K-S.

        cdf : fonction de répartition théorique (par défaut, uniforme [0,1]).
        """
        def uniform_cdf(x: NDArray) -> NDArray:
            return np.clip(x, 0, 1)
        self.cdf = cdf if cdf is not None else uniform_cdf

    def __str__(self) -> str:
        return "K-S_Test"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test de Kolmogorov-Smirnov.
        """
        self._validate_input(data)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        Fn = (np.arange(n) + 1) / n
        Fm = np.arange(n) / n
        Ft = self.cdf(sorted_data)
        D_plus = np.max(Fn - Ft)
        D_minus = np.max(Ft - Fm)
        D = max(D_plus, D_minus)
        ks_stat = D * np.sqrt(n)
        p_value = 1 - kstwobign.cdf(ks_stat)
        crit = kstwobign.ppf(1 - alpha)
        result = {'stat_obs': ks_stat,
                  'stat_crit': crit,
                  'p_value': p_value,
                  'accept': p_value > alpha}
        if info:
            return result, {'n': n, 'D_plus': D_plus, 'D_minus': D_minus}
        return result

class MaximumTest(Tests):
    """
    Test des maximums : compare la distribution des maxima de groupes
    de taille t à la loi X_max^t (i.i.d. uniformes).
    """
    def __init__(self, t: int = 5):
        """Initialise le test des maximums."""
        if t < 2:
            raise ValueError("t doit être au moins 2")
        self.t = t

    def __str__(self) -> str:
        return f"Maximum_Test(t={self.t})"

    def test(self,
             data: NDArray,
             alpha: float = 0.05,
             info: bool = False) -> Union[Dict, Tuple[Dict]]:
        """
        Exécute le test des maximums sur `data`.
        """
        self._validate_input(data)
        n = len(data)
        if n < self.t:
            raise ValueError(f"La taille de l'échantillon ({n}) doit être au moins égal à t ({self.t})")
        # Grouper les données par t
        groups = [data[i:i + self.t] for i in range(0, n - self.t + 1, self.t)]
        max_values = np.array([np.max(g) for g in groups])

        def max_cdf(x):
            """Fonction de répartition des maxima."""
            return x ** self.t

        kstest = KSTest(cdf=max_cdf)
        result = kstest.test(max_values, alpha=alpha, info=info)
        if info :
            result[1].update({'max_df': self.t - 1,
                              'max_values': max_values,
                              't': self.t})
        return result
