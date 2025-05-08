from Tests import *
from

class PokerTest(Tests):
    """
    Test du Poker amélioré avec gestion optimisée des motifs.

    Args:
            group_size: Taille des groupes à analyser (défaut: 5)
    """

    def __init__(self, group_size: int = 5)->None:
        """
        Initialise le test du Poker.
        """
        if group_size < 2:
            raise ValueError("La taille des groupes doit être au moins 2")
        self.group_size = group_size

        self._motifs = generate_motifs(self.group_size)

    def __str__(self) -> str:
        return f"PokerTest(t = {self.group_size}) "

    @lru_cache(maxsize = None)
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
             info: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:

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
        p_value = 1 - chi2.cdf(chi2_stat, df)
        critical_value = chi2.ppf(1 - alpha, df)

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
