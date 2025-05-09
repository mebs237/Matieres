"""
    Point d'entrée principal pour exécuter les tests statistiques sur les générateurs de nombres pseudo-aléatoires.
"""
# IMPORTATION DES MODULES
from numpy.typing import NDArray
import numpy as np
from Generators import PythonGenerator, OurGenerator
from Tests import (Chi2Test , GapTest , KSTest  , MaximumTest , PokerTestv2 , CouponCollectorTest)
from Analysing import ( analyse_sequence )



def read_decimal_file(file_path: str) -> NDArray:
    """
    Lit un fichier contenant les décimales et extrait tous les chiffres.

    Args:
        file_path (str): Chemin vers le fichier à lire.

    Returns:
        Ndarray: Tableau contenant tous les chiffres.
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
            return np.array([int(d) for d in digits], dtype=int)
        else:
            return np.array([], dtype=int)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        return np.array([])
    except ValueError as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return np.array([])


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

    return np.array(result)


def main():
    """
    programme pour l'étude automatique des décimales de e  et la comparaison de notre générateur(OurGenerator) contre celui de python (PythonGenerator)

    """

    #---------------------------------------------------------------
    # ETUDE DU CARACTERE PSEUO-ALEATOIRE UNIFORME DES DECIMALS DE E
    #---------------------------------------------------------------
    # Initialisation de la séquence des décimals de e

    decimals = read_decimal_file("data/e2M.txt")
    seq_ogen = OurGenerator(decimals=decimals).generate(1000)
    seq_ogen = np.array(seq_ogen , dtype=float)
    tes = [GapTest() ,
           Chi2Test() ,
           KSTest() ,
           MaximumTest() ,
           PokerTestv2()]
    resogen  = analyse_sequence(seq_ogen ,
                                 tests=tes,
                                 name = "seq_ogen")
    resogen.save_all('Analyse_results')

    """
    # Analyse  globale  des parties  des décimals avec plusierus tests
    part = [100 , 1000 , 10_000 , 100_000 , 1000_000 , 2000_000]
    for g in part :
        seq = decimals[:g]
        rese = analyse_sequence(seq ,
                                 tests=[GapTest() , Chi2Test() , KSTest() , MaximumTest()],
                                 name = f"e_{g}")
        rese.save_hist('Analyse_results')"""


    #----------------------------
    # COMPARAISON DES GENRATEURS
    #----------------------------

if __name__ == "__main__":
    main()