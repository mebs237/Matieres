"""
Module des fonctions auxiliares à l'implémentation des tests statistiques

Fonctions
---------
- Stirling
- get_optimal_bins
- group_low_frequence
- partitions
- generate_motifs
-

"""


from typing import Tuple, Optional , Dict , Union
from functools import  lru_cache
from collections import Counter
import numpy as np
from numpy.typing import NDArray
from numpy import array, where, argmin, delete, full, unique, bincount, issubdtype, histogram , integer


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

    return k * stirling(n - 1, k) + stirling(n - 1, k - 1)


def partitions(n:int, max_part:Optional[int]=None):
    """
    énumère toutes les façons de découper l'entier n en parts(en d'autres entiers tel que la somme de ce entiers = n) sous forme de listes de taille décroissantes

    Parameters
    ----------
    n : int
        nombre à partitionner
    max_part : _type_, optional
        _description_, by default None

    Yields
    ------
    list
        ensembles des listes partitionant n

    Exemple
    -------
        pour n = 4 , génère successivement ``[4]`` ,  ``[3,1]`` , ``[2,2]`` , ``[2,1,1]`` , ``[1,1,1,1]``

    """
    if n == 0:
        yield []
    else:
        # on ne prend pas de part plus grande que n
        if max_part is None or max_part > n:
            max_part = n
        # on essaie toutes les tailles de part k = max_part, max_part-1, …, 1
        for k in range(max_part, 0, -1):
            # on découpe le reste n-k
            for rest in partitions(n - k, k):
                yield [k] + rest


def name_from_config(config)-> str:
    """
    permet nommer la configuration ou motif des d'éléments d'un groupe ,

    Parameters
    ----------
    config : dict
        dictionnaire des  ``{k : count}`` associant une taille de paquet ``k``  au nombre ``count`` de paquets de cette taille

    Returns
    -------
    str :
        nom de la configuration selon la nomenclature count-k uplet pour dire ``count`` paquet de ``k`` éléments (identiques içi)
        si config est vide : retourn "all different"

    Exemple
    -------
    pour ``config = {}`` : "all different"
    pour ``config = { 3:2 , 2:5}`` -> liste ``[(3,2),(2,1)]`` -> ``["2–3–uplet","1–2–uplet"]`` → "2–3 uplet, 1–2 uplet"
    """
    # Pour chaque (k: count) on fabrique un morceau "count–k–uplet"
    parts = [f"{count}–{k}–uplet" for k, count in sorted(config.items(), reverse=True)]
    # On joint par des virgules, ou on renvoie "all different" si config vide
    return ", ".join(parts) if parts else "all different"

def generate_poker_names(t: int) -> Dict[Tuple[int, ...] , str]:
    """
    Génère le dictionnaire des noms de motifs dans le cas où la taille t entre 2 et 5
    """
    base_names = {
        2: {
            (2,): "paire",
            (1,1): "all different"
        },
        3: {
            (3,): "brelan",
            (2,1): "paire",
            (1,1,1): "all different"
        },
        4: {
            (4,): "carré",
            (3,1): "brelan",
            (2,2): "double paire",
            (2,1,1): "paire",
            (1,1,1,1): "all different"
        },
        5: {
            (5,): "quintuplet",
            (4,1): "carré",
            (3,2): "full",
            (3,1,1): "brelan",
            (2,2,1): "double paire",
            (2,1,1,1): "paire",
            (1,1,1,1,1): "all different"
        }
    }
    return base_names.get(t, {})

def generate_motifs(t: int) -> Dict[str,Dict[int,int]]:
    """
    Construit un dictionnaire de tous les motifs possibles pour un groupe de taille t.
    """
    mapping : Dict[str,Dict[int,int]]= {}

    poker_names = generate_poker_names(t)

    for p in partitions(t): # génère toutes les partitions de t

        cfg = Counter(p)

        freq = tuple(sorted(p, reverse=True))

        if freq == (t,) :
            name = "all identic"
        elif 2 <= t <= 5 :
            name = poker_names.get(freq, name_from_config(cfg))
        else:
            name = name_from_config(dict(cfg))

        if name in mapping and mapping[name]!=dict(cfg):
            raise ValueError(f" Nom dupliqué pour des configurations différentes : {name}")
        mapping[name] = dict(cfg)

    return mapping

def invert_dict(motifs: Dict[str, Dict[int, int]]) -> Dict[Tuple[Tuple[int, int], ...], str]:
    """
    Inverse le dictionnaire des motifs en de dictionnaire { nom_config : config }  en {config : nom_motif }
    """
    invert = {}
    for name , cfg in motifs.items():
        key = tuple(sorted(cfg.items()))
        invert[key] = name
    return invert

def assign_motifs(group:NDArray,motifs : Dict[str,Dict[int,int]]) -> str:
    """
    Assigne un group de taille t au motif correspondant

    Parameters
    ----------
    group : NDArray
        groupe d'éléments de taille t
    motifs : Dict[str,Dict[int,int]]
        motifs possibles pour un  groupes de taille t

    Returns
    -------
    str
        nom du motif auquel correspond le group
    """

    freq = Counter(group)
    cfg = Counter(freq.values())
    key = tuple(sorted(cfg.items()))
    return invert_dict(motifs).get(key , "inconnu")

def compute_bins(data:NDArray,
                 n_bins:int,
                 method:str="quantile",
                 a:float = 0,
                 b:float = 1
                 )->NDArray:
    pass

def get_optimal_bins(data: NDArray,
                      k: int = 10,
                      probabilities: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
    """
    Calcule ``le`` nombre optimal ``k`` de classes/bins optimaux pour les données.
    Utilise l' `algorithme de Sturges <https://fr.wikipedia.org/wiki/Règle_de_Sturges>`_ pour les données continues.

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
    data = array(data)
    n = len(data)

    if n == 0:
        raise ValueError("Les données ne peuvent pas être vides")

    if issubdtype(data.dtype, integer):
        # Pour les entiers, utiliser les valeurs uniques
        unique_values = unique(data)
        d = len(unique_values)
        if d <= k:
            observed = bincount(data, minlength=max(data) + 1)
            if probabilities is None:
                expected = full(d, n / len(unique_values))
            else:
                # Utiliser les probabilités fournies
                probabilities = array(probabilities)
                if len(probabilities) != len(unique_values):
                    raise ValueError("Le nombre de probabilités doit correspondre au nombre de valeurs uniques")
                expected = n * probabilities
            return observed, expected

    # Pour les données continues, utiliser l'algorithme de Sturges
    k = min(k, int(1 + np.log2(n)))
    observed , _ = histogram(data, bins=k)

    if probabilities is None:
        # Distribution uniforme
        expected = full(k, n / k)
    else:
        # Utiliser les probabilités fournies
        probabilities = array(probabilities)
        if len(probabilities) != k:
            raise ValueError("Le nombre de probabilités doit correspondre au nombre de bins")
        expected = n * probabilities

    return observed, expected



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
            # Sinon, regrouper avec la classe adjacente ayant l'effectif attendu minimal
            if expected_new[idx_min-1] <= expected_new[idx_min+1]:
                idx_adjacent = idx_min - 1
            else:
                idx_adjacent = idx_min + 1

        # Regrouper les classes à gauche
        observed_new[min(idx_min, idx_adjacent)] += observed_new[max(idx_min, idx_adjacent)]
        expected_new[min(idx_min, idx_adjacent)] += expected_new[max(idx_min, idx_adjacent)]

        # Supprimer la classe regroupée
        observed_new = delete(observed_new, max(idx_min, idx_adjacent))
        expected_new = delete(expected_new, max(idx_min, idx_adjacent))

    return observed_new, expected_new


def discretize(data: NDArray ,
               a: Union[int , float] = 0,
               b : Union[int , float] = 1,
               n_bins:Optional[int] = None,
               binning : str = "quantile") -> Tuple[NDArray, int]:
    """
    Discrétisation des éléments d'une séquence pour obtenir des classes ou catégories

    Parameters
    ----------
    data : séquence à diviser
    a : borne inférieure de l'intervalle des valeurs de data
    b : brone supérieure de l'intervalle des valeurs de data
    n_bins : nombre de catégories/intervalles désirés
    bining : méthode de division/discrétisation

    Fonctionnement
    ---------
    - si data est une liste d'entiers , on prend juste l'ensemble des valeurs unique
    - sinon ( c'est a dire contient des valeurs continues) , en supposant que ``[a,b]`` soit la plage de ces valeurs ;
        * on divise la plage en ``n_bins`` intervalles selon la méthode ``binning`` demandée par quantille ( binning = quantile ) ou par intervalle fixe (binning = uniform)
        * à chaque valeur de data , on associe l'indice/index de l'intevalle auquel il appratient


    Returns
    -------
        discretize (NDArray):  ensemble des catégories/classes
        d (int):  nombres de catégoris/classes
    """
    data = np.array(data)

    if np.issubdtype(data.dtype, np.integer):
        return data, len(np.unique(data))


    if n_bins is None:
        d = int(1 + np.log10(len(data)))  # règle de Sturges
    d=n_bins

    if binning == "quantile":
        bins = np.quantile(data, np.linspace(a,b,d + 1)[1:-1])
    elif binning == "uniform":
        bins = np.linspace(np.min(data), np.max(data), d + 1)[1:-1]
    else:
        raise NotImplementedError("méthodes de division ( ``binning``) non pris en charge , essayez \"quantile\" ou \"uniform\" ")
    discretized = np.digitize(data, bins)
    return discretized, d


def process_bins(data: NDArray,
                 k: Optional[int] = None,
                 probabilities: Optional[NDArray] = None,
                 binning: str = "uniform",
                 return_counts: bool = False
                 ) -> Union[Tuple[NDArray, int], Tuple[NDArray, int, NDArray, NDArray]]:
    """
    Traite les données pour les discrétiser en bins/classes et optionnellement calculer les effectifs observés/attendus.

    Parameters
    ----------
    data:
        Données à traiter.
    k:
        Nombre de bins/classes souhaité (défaut: règle de Sturges).
    probabilities:
        Probabilités théoriques pour chaque classe (défaut: distribution uniforme).
    binning:
        Méthode de discrétisation ('quantile', 'uniform', ou 'auto').
    return_counts:
        Si True, retourne également les effectifs observés et attendus.

    Returns
    -------
    Tuple:
    - discretized (NDArray):
        Données discrétisées.
    - d (int):
        Nombre de classes/bins.
    - observed (NDArray , Optionnal):
        Effectifs observés pour chaque classe/bin.
    - expected (NDArray , optionnal):
        Effectifs attendus pour chaque classe/bin.
    """
    data = np.array(data)
    n = len(data)
    # Cas des données entières
    if np.issubdtype(data.dtype, np.integer):
        unique_values = np.unique(data)
        d = len(unique_values)
        observed = np.bincount(data, minlength=d)
        if probabilities is None :
            expected = np.full(d, n / d)
        else :
            expected = n * np.array(probabilities)
            if len(probabilities)!= d:
                raise ValueError("Le nombre de probabilités ne correspond pas au nombred de classes")

        return (discretized, d, observed, expected) if return_counts else (discretized, d)

    # Cas des données continues
    n = len(data)
    if k is None:
        d = int(1 + np.log10(n))  # Règle de Sturges
    else:
        d = k

    if binning == "quantile":
        bins = np.quantile(data, np.linspace(0, 1, d + 1))
    elif binning == "uniform":
        bins = np.linspace(np.min(data), np.max(data), d + 1)
    else:
        raise ValueError("Méthode de discrétisation non prise en charge. Utilisez 'quantile' ou 'uniform'.")

    discretized = np.digitize(data, bins[:-1]) - 1  # Associer chaque valeur à un bin
    observed, _ = np.histogram(data, bins=bins)
    expected = np.full(d, n / d) if probabilities is None else n * np.array(probabilities)

    return (discretized, d, observed, expected) if return_counts else (discretized, d)