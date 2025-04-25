"""
    Point d'entrée principal pour exécuter les tests statistiques sur les générateurs de nombres pseudo-aléatoires.
"""
# IMPORTATION DES MODULES
from Generators import PythonGenerator, OurGenerator , Generators
from Tests import (Chi2Test , GapTest , KSTest , PokerTest , MaximumTest , CouponCollectorTest , Tests)
from Analysing import ( analyse_sequence , evaluate_generator , analyse_convergence , AnalysisResult )
from numpy.typing import NDArray
import numpy as np



def read_decimal_file(file_path: str) -> NDArray:
    """
    Lit un fichier contenant des nombres décimaux, extrait les chiffres après la virgule,
    et les stocke dans un tableau.

    Args:
        file_path (str): Chemin vers le fichier à lire.

    Returns:
        np.ndarray: Tableau contenant les chiffres après la virgule.
    """
    try:
        # Lire le fichier ligne par ligne
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extraire les chiffres après la virgule pour chaque ligne
        decimals = []
        for line in lines:
            line = line.strip()  # Supprimer les espaces ou sauts de ligne
            if '.' in line:  # Vérifier si la ligne contient un nombre décimal
                _ , fractional_part = line.split('.', 1)  # Séparer la partie entière et la partie décimale
                decimals.extend(list(fractional_part))  # Ajouter chaque chiffre à la liste

        # Convertir en tableau numpy d'entiers
        return np.array([int(digit) for digit in decimals], dtype=int)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        return np.array([])
    except ValueError as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return np.array([])


def main():
    """
    programme pour l'étude automatique des décimales de e  et la comparaison de notre générateur(OurGenerator) contre celui de python (PythonGenerator)

    """



    #---------------------------------------------------------------
    # ETUDE DU CARACTERE PSEUO-ALEATOIRE UNIFORME DES DECIMALS DE E
    #---------------------------------------------------------------
    # Initialisation de notre

    decimals = read_decimal_file("e2M.txt")



    gen1 = PythonGenerator()
    gen2 = OurGenerator(data,name="e_Genarator")
    tests = [Chi2Test(), GapTest() , KS()]

    # Analyser la convergence
    conv_analysis1 = analyse_convergence(gen1, tests)
    conv_analysis2 = analyse_convergence(gen2, tests)

    # Comparer les générateurs
    comparison = compare_generators_score(gen1, gen2, tests)

    print(f"Score final : Gen1 {comparison['gen1']} - {comparison['gen2']} Gen2")

    return conv_analysis1, conv_analysis2, comparison

if __name__ == "__main__":
    main()