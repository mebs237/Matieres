"""
Module utilitaire pour les implémentations des tests d'hypothèse statistique .
"""


from typing import Tuple, List, Union , Optional
from functools import cache
import numpy as np
from numpy.typing import NDArray
from numpy import array, where, argmin, delete, full, unique, bincount, issubdtype, histogram , integer , floating

# Type générique pour les nombres (entiers ou décimaux)
Number = Union[int , float , integer , floating]
Array = Union[List[Number],NDArray[Number]]

# Les fonctions communes à plusieurs tests
@cache
def stirling(n: int, k: int) -> int:
    """
    Méthode pour calculer le nombre de stirling de deuxième espèce.
    """
    assert n >= 0 and k >= 0, "n et k doivent être plus grand que 0."

    if n == k:
        return 1

    elif n == 0 or k == 0:
        return 0

    return k * stirling(n - 1, k) + stirling(n - 1, k - 1)


def _get_optimal_bins(data: Array,
                      k: int = 10,
                      probabilities: Optional[Array] = None) -> Tuple[NDArray[Number], NDArray[Number]]:
    """
    Calcule les bins optimaux pour les données.
    Utilise l'algorithme de Sturges pour les données continues.

    Args:
        data: Données à regrouper en bins
        k: Nombre de bins souhaité (défaut: 10)
        probabilities: Probabilités théoriques pour chaque bin (défaut: None, distribution uniforme)

    Returns:
        Tuple (observed, expected) contenant:
            - observed: Effectifs observés dans chaque bin
            - expected: Effectifs attendus dans chaque bin
    """
    # Convertir en numpy array si nécessaire
    data = array(data)
    n = len(data)

    if n == 0:
        raise ValueError("Les données ne peuvent pas être vides")

    if issubdtype(data.dtype, integer):
        # Pour les entiers, utiliser les valeurs uniques
        unique_values = unique(data)
        if len(unique_values) <= k:
            observed = bincount(data, minlength=max(data) + 1)
            if probabilities is None:
                expected = full(len(unique_values), n / len(unique_values))
            else:
                # Utiliser les probabilités fournies
                probabilities = array(probabilities)
                if len(probabilities) != len(unique_values):
                    raise ValueError("Le nombre de probabilités doit correspondre au nombre de valeurs uniques")
                expected = n * probabilities
            return observed, expected

    # Pour les données continues, utiliser l'algorithme de Sturges
    k = min(k, int(1 + 3.322 * np.log10(n)))
    hist, bin_edges = histogram(data, bins=k)

    if probabilities is None:
        # Distribution uniforme
        expected = full(k, n / k)
    else:
        # Utiliser les probabilités fournies
        probabilities = array(probabilities)
        if len(probabilities) != k:
            raise ValueError("Le nombre de probabilités doit correspondre au nombre de bins")
        expected = n * probabilities

    return hist, expected

def group_low_frequence(observed: Array,
                        expected: Array,
                        seuil: int = 5,
                        mseuil: int = 1) -> Tuple[NDArray[Number], NDArray[Number]]:
    """
    Regroupe les classes avec des effectifs faibles selon une stratégie spécifique.

    Stratégie de regroupement:
    - Si deux classes adjacentes ont des effectifs faibles (< seuil ou < mseuil), on les regroupe
    - Sinon, on regroupe la classe faible avec l'une des classes adjacentes ayant l'effectif minimum

    Args:
        observed: Liste des effectifs observés
        expected: Liste des effectifs attendus
        seuil: Seuil principal pour les effectifs (défaut: 5)
        mseuil: Seuil minimal pour les effectifs (défaut: 1)

    Returns:
        Tuple (observed_new, expected_new) contenant les nouvelles listes regroupées
    """
    # Convertir en numpy arrays si nécessaire
    observed = array(observed)
    expected = array(expected)

    # Vérifier que les listes ont la même taille
    if len(observed) != len(expected):
        raise ValueError("Les listes observed et expected doivent avoir la même taille")

    # Vérifier que les listes ne sont pas vides
    if len(observed) == 0:
        return observed, expected

    # Copier les listes pour ne pas modifier les originales
    observed_new = observed.copy()
    expected_new = expected.copy()

    # Fonction pour vérifier si les conditions sont respectées
    def conditions_respectees():
        # Compter les classes avec effectif >= seuil
        classes_suffisantes = np.sum(observed_new >= seuil)
        # Compter les classes avec effectif > mseuil
        classes_minimales = np.sum(observed_new > mseuil)
        # Total des classes
        total_classes = len(observed_new)

        # Vérifier les conditions: 80% >= seuil et 100% > mseuil
        return (classes_suffisantes / total_classes >= 0.8 and
                classes_minimales == total_classes)

    # Tant que les conditions ne sont pas respectées et qu'il y a plus d'une classe
    while not conditions_respectees() and len(observed_new) > 1:
        # Trouver les indices des classes avec effectif faible
        indices_faibles = where((observed_new < seuil) & (observed_new <= mseuil))[0]

        if len(indices_faibles) == 0:
            # S'il n'y a plus de classes faibles mais que les conditions ne sont pas respectées,
            # on cherche les classes avec effectif minimal
            indices_faibles = where(observed_new <= mseuil)[0]

        if len(indices_faibles) == 0:
            # S'il n'y a plus de classes à regrouper, on sort de la boucle
            break

        # Prendre le premier indice faible
        idx = indices_faibles[0]

        # Déterminer l'indice adjacent avec lequel regrouper
        if idx == 0:
            # Si c'est la première classe, regrouper avec la suivante
            idx_adjacent = 1
        elif idx == len(observed_new) - 1:
            # Si c'est la dernière classe, regrouper avec la précédente
            idx_adjacent = idx - 1
        else:
            # Sinon, regrouper avec la classe adjacente ayant l'effectif minimum
            if observed_new[idx-1] <= observed_new[idx+1]:
                idx_adjacent = idx - 1
            else:
                idx_adjacent = idx + 1

        # Regrouper les classes
        observed_new[min(idx, idx_adjacent)] += observed_new[max(idx, idx_adjacent)]
        expected_new[min(idx, idx_adjacent)] += expected_new[max(idx, idx_adjacent)]

        # Supprimer la classe regroupée
        observed_new = delete(observed_new, max(idx, idx_adjacent))
        expected_new = delete(expected_new, max(idx, idx_adjacent))

    return observed_new, expected_new


def group_low_frequence_chi2(observed: Array,
                          expected: Array,
                          min_expected: float = 5.0) -> Tuple[Array, Array]:
    """
    Regroupe les classes pour le test du chi-carré selon la règle standard.

    Args:
        observed: Liste des effectifs observés
        expected: Liste des effectifs attendus
        min_expected: Effectif minimal attendu par classe (défaut: 5.0)

    Returns:
        Tuple (observed_new, expected_new) contenant les nouvelles listes regroupées
    """
    # Convertir en numpy arrays si nécessaire
    observed = array(observed)
    expected = array(expected)

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
        idx_min = argmin(expected_new)

        # Déterminer l'indice adjacent avec lequel regrouper
        if idx_min == 0:
            # Si c'est la première classe, regrouper avec la suivante
            idx_adjacent = 1
        elif idx_min == len(expected_new) - 1:
            # Si c'est la dernière classe, regrouper avec la précédente
            idx_adjacent = idx_min - 1
        else:
            # Sinon, regrouper avec la classe adjacente ayant l'effectif attendu minimum
            if expected_new[idx_min-1] <= expected_new[idx_min+1]:
                idx_adjacent = idx_min - 1
            else:
                idx_adjacent = idx_min + 1

        # Regrouper les classes
        observed_new[min(idx_min, idx_adjacent)] += observed_new[max(idx_min, idx_adjacent)]
        expected_new[min(idx_min, idx_adjacent)] += expected_new[max(idx_min, idx_adjacent)]

        # Supprimer la classe regroupée
        observed_new = delete(observed_new, max(idx_min, idx_adjacent))
        expected_new = delete(expected_new, max(idx_min, idx_adjacent))

    return observed_new, expected_new


t : Array = [1,2,6,59]
print(type(t))
