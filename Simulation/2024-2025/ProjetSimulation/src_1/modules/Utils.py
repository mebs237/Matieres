"""
Modules de fonctions utilitaires
"""
# IMPORTATION DES MODULES
from functools import wraps
from typing import Union , List
from numpy.typing import NDArray
from numpy import array


def read_decimal_file(file_path: str) -> NDArray:
    """
    Lit un fichier contenant les décimales et extrait tous les chiffres.

    Args:
        file_path (str): Chemin vers le fichier à lire.

    Returns:
        NDArray: Tableau contenant tous les chiffres.
    """
    try:
        # Lecture et nettoyage du fichier
        with open(file_path, 'r', encoding='utf-8') as file:
            content = ''.join(file.readlines())
            # Supprimer les espaces, retours à la ligne et commentaires
            content = ''.join(c for c in content if c.isdigit() or c == '.')

        # Trouver le point décimal et extraire tous les chiffres après
        if '.' in content:
            _, digits = content.split('.', 1)
            return array([int(d) for d in digits], dtype=int)
        else:
            return array([], dtype=int)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        return array([])
    except ValueError as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return array([])


def group_digits(array_1: NDArray, k: int) -> NDArray:
    """
    Groupe les chiffres d'un tableau en nombres de k chiffres consécutifs.

    Parameters
    ----------
    array_1 : NDArray
        Tableau de chiffres à grouper
    k : int
        Nombre de chiffres consécutifs par groupe

    Returns
    -------
    NDArray
        Tableau contenant les nombres formés par k chiffres consécutifs

    Examples
    --------
    >>> group_digits(np.array([1,2,3,4,5,6,7]), 3)
    array([123, 456, 7])
    >>> group_digits(np.array([1,2,3,4,5]), 2)
    array([12, 34, 5])
    """
    if k <= 0:
        raise ValueError("k doit être strictement positif")

    n = len(array_1)
    # Initialiser le tableau résultat
    result = []

    # Traiter les groupes complets de k chiffres
    i = 0
    while i <= n - k:
        # Convertir k chiffres en un seul nombre
        number = sum(array_1[i + j] * (10 ** (k - j - 1)) for j in range(k))
        result.append(number)
        i += k

    # Traiter les chiffres restants s'il y en a
    if i < n:
        remaining_digits = n - i
        number = sum(array_1[i + j] * (10 ** (remaining_digits - j - 1))
                    for j in range(remaining_digits))
        result.append(number)

    return array(result)

def requires__run(method):
    """décorateur pour savoir si les calculs ont déjà été effectués"""
    @wraps(method)
    def wrapper(self,*args,**kwargs):
        if not getattr(self , "runed",False):
            raise RuntimeError(f"vous devez exécutez `.run()` avant d'appeler {method.__name__}")
        return method(self,*args,**kwargs)
    return wrapper

def to_wind_size(granularities: List[Union[int,float]],
                 len_seq: int
                 )->List[int]:
    """
    Transforme une granularité ou liste de granularités en taille(s) entière(s) de sous séquence
    """
    if len_seq<=0:
        raise ValueError("la taillle de la séquence doit être > 0")

    # Vérifier si les granularités sont des entiers ou des fractions
    if all([isinstance(g,int) and 0<g<=len_seq for g in granularities]) :
        return granularities
    elif all([isinstance(g,float) and 0.0<g<=1.0 for g in granularities]):
        return [int(g*len_seq) for g in granularities]
    else :
        raise ValueError(f"les granularités doivent être > 0 ,  être soit des entiers <={len_seq} , soit des fractions  <=1 de la taille de ")

