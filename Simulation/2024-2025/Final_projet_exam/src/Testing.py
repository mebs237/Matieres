# import multiprocessing.pool
import numpy as np
from Tests import Chi2Test, GapTest, Test
from Generators import Generators

def test_generator(generator: Generators, alpha: float = 0.05, k : int = 100, n: int = 200000, tests: list[Test] = None,display =False, plot = False ):
    """
    Teste des générateurs avec plusieurs tests d'hypothèses.

    Args:
        generator:Les générateurs à tester.
        alpha: Le seuil utilisé pour les tests.
        k:Nombre d'itérations.
        n:Nombre de nombres à générer par itération.
        tests:Liste des tests d'hypothèses à appliquer.
        dsiplay : pour afficher ou pas les resultats en invite de commande

    Returns:
        si plot vrai retourne un dictionnaire 'test' : p-valeur , sinon juste
    """
    #assert isinstance(generator , Generator)

    if tests is None:
        tests = [Chi2Test(), GapTest()]

    p_values = np.zeros(len(tests))
    test_num_success = np.zeros(len(tests))

    #on va générer n nombres k fois
    for _ in range(k):
        generated_numbers = generator.generate_n(n)

        for i, test in enumerate(tests):
            result = test.test(generated_numbers)
            p_value = result['p_value']
            p_values[i] += p_value
            if p_value > alpha:
                test_num_success[i] += 1

    # Calcul des moyennes
    p_values /= k
    success_rates = test_num_success / k

    # Affichage des résultats
    if display :
        print(generator)
        max_length = max(len(str(test)) for test in tests)
        print(f"| {'Tests':^{max_length}} | Taux de succès | p-valeur moyennes |")
        print(f"|{'-' * (max_length + 35)}|")

        for i, test in enumerate(tests):
            print(f"| {str(test):^{max_length}} | {success_rates[i]:^14.1%} | {p_values[i]:^9.3f} |")

    if not plot:
        return test_num_success, p_values
    else :
        plot_hist= {(generator,success_rates): (name,value ) for name,value in zip(tests,p_values)}
        return plot_hist


def test_decimals(decimals: np.ndarray, alpha: float = 0.05, tests: list[Test] = None,display = False , hist = False)->np.ndarray:
    """
    Teste les décimales de pi avec plusieurs tests d'hypothèses.

    Args:
        decimals: la suite de nombres à tester.
        alpha: Le seuil utilisé pour le test.
        tests: Liste des tests d'hypothèses à appliquer.

    Returns:

    """
    if tests is None:
        tests = [Chi2Test(),GapTest()]


    p_values = np.zeros(len(tests))
    test_success = np.zeros(len(tests), dtype=bool)

    for i, test in enumerate(tests):
        result = test.test(decimals)
        p_values[i] = result['p_value']
        test_success[i] = p_values[i] > alpha

    #affichage
    if display :
        max_length = max(len(str(test)) for test in tests)
        print(f"| {'Tests':^{max_length}} | Test réussi | p-valeur|")
        print(f"|{'-' * (max_length + 34)}|")

        for i, test in enumerate(tests):
            print(f"| {str(test):^{max_length}} | {'Oui' if test_success[i] else 'Non':^11} | {p_values[i]:^9.3f} |")

    return p_values

