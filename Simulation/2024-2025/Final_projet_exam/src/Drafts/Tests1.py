import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union
from scipy import stats
import math

class StatisticalTests:
    def __init__(self):
        pass
    
    def chi_square_test(self, data: NDArray, bins: int = 10, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        """
        Test du Chi-carré pour l'uniformité d'une distribution.
        
        Args:
            data: Séquence de nombres à tester
            bins: Nombre de catégories/bins pour la discrétisation
            alpha: Niveau de signification
            info: Si True, retourne un dictionnaire avec les informations détaillées
            
        Returns:
            p-value ou dictionnaire d'informations du test
        """
        # Déterminer le nombre d'observations et la fréquence attendue par bin
        n = len(data)
        expected_freq = n / bins
        
        # Pour les entiers, on utilise les valeurs directement si la plage est limitée
        if np.issubdtype(data.dtype, np.integer):
            if np.max(data) - np.min(data) + 1 <= bins:
                # Utiliser chaque valeur entière comme une catégorie
                unique_values, observed_freq = np.unique(data, return_counts=True)
                df = len(unique_values) - 1
                # Calculer les fréquences attendues
                expected_freqs = np.full(len(unique_values), expected_freq)
            else:
                # Discrétiser en bins
                hist, _ = np.histogram(data, bins=bins)
                observed_freq = hist
                df = bins - 1
                expected_freqs = np.full(bins, expected_freq)
        else:
            # Pour les nombres à virgule flottante, discrétiser en intervalles
            hist, _ = np.histogram(data, bins=bins)
            observed_freq = hist
            df = bins - 1
            expected_freqs = np.full(bins, expected_freq)
        
        # Calculer la statistique du chi-carré
        chi2_stat = np.sum((observed_freq - expected_freqs) ** 2 / expected_freqs)
        
        # Calculer la p-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        # Calculer la valeur critique
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
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
    
    def gap_test(self, data: NDArray, alpha: float = 0.05, alpha_gap: float = 0.0, beta_gap: float = 0.5, max_gap: int = 10, n_gaps: int = 30, info: bool = False) -> Union[float, Dict]:
        """
        Test du gap pour analyser la longueur des intervalles entre les occurrences de valeurs dans [alpha_gap, beta_gap).
        
        Args:
            data: Séquence de nombres à tester (généralement des nombres à virgule flottante dans [0, 1))
            alpha: Niveau de signification
            alpha_gap: Borne inférieure de l'intervalle d'intérêt
            beta_gap: Borne supérieure de l'intervalle d'intérêt
            max_gap: Longueur maximale de gap à considérer explicitement
            n_gaps: Nombre de gaps à observer
            info: Si True, retourne un dictionnaire avec les informations détaillées
            
        Returns:
            p-value ou dictionnaire d'informations du test
        """
        # Vérifier que les données sont dans [0, 1)
        if np.min(data) < 0 or np.max(data) >= 1:
            raise ValueError("Le test du gap s'applique à des données dans l'intervalle [0, 1)")
        
        # Calculer la probabilité p = beta_gap - alpha_gap
        p = beta_gap - alpha_gap
        
        # Initialiser le compteur de gaps
        gap_counts = np.zeros(max_gap + 1, dtype=int)
        
        # Variables pour parcourir la séquence
        gaps_found = 0
        current_gap = 0
        i = 0
        
        # Parcourir la séquence pour trouver les gaps
        while gaps_found < n_gaps and i < len(data):
            if alpha_gap <= data[i] < beta_gap:
                # Enregistrer la longueur du gap
                if current_gap >= max_gap:
                    gap_counts[max_gap] += 1
                else:
                    gap_counts[current_gap] += 1
                gaps_found += 1
                current_gap = 0
            else:
                current_gap += 1
            i += 1
        
        # Vérifier si nous avons trouvé assez de gaps
        if gaps_found < n_gaps:
            raise ValueError(f"Nombre de gaps insuffisant. Trouvé {gaps_found}, requis {n_gaps}")
        
        # Calculer les probabilités théoriques
        probs = np.zeros(max_gap + 1)
        for i in range(max_gap):
            probs[i] = p * ((1 - p) ** i)
        probs[max_gap] = (1 - p) ** max_gap
        
        # Calculer la statistique du chi-carré pour les longueurs de gap
        expected_counts = probs * n_gaps
        chi2_stat = np.sum((gap_counts - expected_counts) ** 2 / expected_counts)
        
        # Calculer les degrés de liberté et la p-value
        df = max_gap
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        # Calculer la valeur critique
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        if info:
            return {
                "statistic": chi2_stat,
                "critical_value": critical_value,
                "p_value": p_value,
                "df": df,
                "reject_null": p_value < alpha,
                "gap_counts": gap_counts,
                "expected_counts": expected_counts
            }
        else:
            return p_value
    
    def poker_test(self, data: NDArray, group_size: int = 5, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        """
        Test du poker pour analyser les motifs dans des groupes d'entiers.
        
        Args:
            data: Séquence d'entiers à tester
            group_size: Taille des groupes (usuellement 5)
            alpha: Niveau de signification
            info: Si True, retourne un dictionnaire avec les informations détaillées
            
        Returns:
            p-value ou dictionnaire d'informations du test
        """
        # Vérifier que les données sont des entiers
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Le test du poker s'applique à des séquences d'entiers")
        
        # Nombre de groupes complets
        n_groups = len(data) // group_size
        
        # Définir les catégories de motifs
        # Pour group_size=5, il y a 7 catégories:
        # 1. Tous différents
        # 2. Une paire
        # 3. Deux paires
        # 4. Trois d'un type
        # 5. Full house
        # 6. Quatre d'un type
        # 7. Cinq d'un type
        
        # Initialiser les compteurs de motifs
        pattern_counts = np.zeros(7, dtype=int)
        
        # Parcourir les groupes
        for i in range(n_groups):
            group = data[i * group_size:(i + 1) * group_size]
            
            # Compter les occurrences de chaque valeur
            values, counts = np.unique(group, return_counts=True)
            
            # Déterminer le motif
            if len(values) == group_size:  # Tous différents
                pattern_counts[0] += 1
            elif len(values) == group_size - 1:  # Une paire
                pattern_counts[1] += 1
            elif len(values) == group_size - 2:
                if np.max(counts) == 2:  # Deux paires
                    pattern_counts[2] += 1
                else:  # Trois d'un type
                    pattern_counts[3] += 1
            elif len(values) == 2:
                if np.max(counts) == 3:  # Full house
                    pattern_counts[4] += 1
                else:  # Quatre d'un type
                    pattern_counts[5] += 1
            else:  # Cinq d'un type
                pattern_counts[6] += 1
        
        # Calculer les probabilités théoriques pour chaque motif
        # Ces probabilités dépendent des nombres distincts possibles dans la séquence (d)
        d = len(np.unique(data))
        
        # Calcul des probabilités théoriques pour group_size=5
        # Ces formules sont spécifiques à group_size=5 et devront être ajustées pour d'autres tailles
        if group_size == 5:
            probs = np.zeros(7)
            
            # Tous différents: d(d-1)(d-2)(d-3)(d-4)/d^5
            probs[0] = (d * (d-1) * (d-2) * (d-3) * (d-4)) / (d**5)
            
            # Une paire: 10 * d(d-1)(d-2)(d-3)/d^5
            probs[1] = 10 * (d * (d-1) * (d-2) * (d-3)) / (d**5)
            
            # Deux paires: 15 * d(d-1)(d-2)/d^5
            probs[2] = 15 * (d * (d-1) * (d-2)) / (d**5)
            
            # Trois d'un type: 10 * d(d-1)(d-2)/d^5
            probs[3] = 10 * (d * (d-1) * (d-2)) / (d**5)
            
            # Full house: 10 * d(d-1)/d^5
            probs[4] = 10 * (d * (d-1)) / (d**5)
            
            # Quatre d'un type: 5 * d(d-1)/d^5
            probs[5] = 5 * (d * (d-1)) / (d**5)
            
            # Cinq d'un type: d/d^5
            probs[6] = d / (d**5)
        else:
            raise ValueError("Le test du poker est actuellement implémenté uniquement pour group_size=5")
        
        # Calculer les fréquences attendues
        expected_counts = probs * n_groups
        
        # Filtrer les catégories avec des fréquences attendues trop faibles (< 5)
        valid_categories = expected_counts >= 5
        
        if np.sum(valid_categories) < 2:
            raise ValueError("Pas assez de catégories valides pour le test du chi-carré")
        
        # Combiner les catégories si nécessaire
        if not np.all(valid_categories):
            # Combiner les catégories non valides avec la catégorie valide la plus proche
            for i in range(len(valid_categories)):
                if not valid_categories[i]:
                    # Trouver l'indice de la catégorie valide la plus proche
                    closest_valid = np.argmin(np.abs(np.arange(len(valid_categories))[valid_categories] - i))
                    # Ajouter le compte à cette catégorie
                    pattern_counts[closest_valid] += pattern_counts[i]
                    pattern_counts[i] = 0
                    expected_counts[closest_valid] += expected_counts[i]
                    expected_counts[i] = 0
        
        # Calculer la statistique du chi-carré en ignorant les catégories non valides
        chi2_stat = np.sum(((pattern_counts[valid_categories] - expected_counts[valid_categories]) ** 2) / expected_counts[valid_categories])
        
        # Calculer les degrés de liberté et la p-value
        df = np.sum(valid_categories) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        # Calculer la valeur critique
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        if info:
            return {
                "statistic": chi2_stat,
                "critical_value": critical_value,
                "p_value": p_value,
                "df": df,
                "reject_null": p_value < alpha,
                "pattern_counts": pattern_counts,
                "expected_counts": expected_counts,
                "valid_categories": valid_categories
            }
        else:
            return p_value
    
    def coupon_collector_test(self, data: NDArray, d: int = None, max_length: int = None, n_segments: int = 30, alpha: float = 0.05, info: bool = False) -> Union[float, Dict]:
        """
        Test du collectionneur de coupons pour analyser le nombre d'éléments nécessaires pour obtenir un ensemble complet de valeurs.
        
        Args:
            data: Séquence d'entiers à tester
            d: Nombre de valeurs distinctes possibles dans la séquence (si None, utilise le nombre de valeurs uniques)
            max_length: Longueur maximale de segment à considérer (si None, utilise 2*d)
            n_segments: Nombre de segments à observer
            alpha: Niveau de signification
            info: Si True, retourne un dictionnaire avec les informations détaillées
            
        Returns:
            p-value ou dictionnaire d'informations du test
        """
        # Vérifier que les données sont des entiers
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Le test du collectionneur de coupons s'applique à des séquences d'entiers")
        
        # Déterminer le nombre de valeurs distinctes
        if d is None:
            d = len(np.unique(data))
        
        # Déterminer la longueur maximale de segment
        if max_length is None:
            max_length = 2 * d
        
        # Initialiser le compteur de segments
        segment_counts = np.zeros(max_length - d + 2, dtype=int)
        
        # Variables pour parcourir la séquence
        segments_found = 0
        i = 0
        
        # Parcourir la séquence pour