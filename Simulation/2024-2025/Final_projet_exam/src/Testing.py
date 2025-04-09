# import multiprocessing.pool

import numpy as np
from numpy import ndarray , zeros
from typing import List, Dict, Tuple, Optional, Union
from tests import *
from Generators import Generator , OurGenerator
import time

def test_generator(generator: Generator,
                  tests: List[StatisticalTest],
                  alpha: List[float] = [0.05],
                  k: int = 100,
                  n: int = 200000,
                  return_history: bool = False,
                  display: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray]]:
    """
    Teste k fois un générateur avec plusieurs tests d'hypothèses et plusieurs seuils.
    Retourne une série de mesures telles que : le pourcentage de réussite aux tests,
    les p-valeurs moyennes pour chaque test, les statistiques observées moyennes, etc.

    Args:
        generator: Le générateur à tester.
        tests: Liste des tests d'hypothèses à appliquer.
        alpha: Liste des seuils utilisés pour les tests.
        k: Nombre d'itérations.
        n: Nombre d'éléments générés à chaque itération.
        return_history: Si True, retourne également l'historique complet des tests.
        display: Si True, affiche les résultats dans la console.

    Returns:
        Si return_history est False: dictionnaire contenant les résultats agrégés.
        Si return_history est True: tuple (dictionnaire de résultats, historique complet).
    """
    if tests is None:
        raise ValueError("La liste des tests ne peut pas être None")

    n_test = len(tests)
    n_seuil = len(alpha)

    # Historique des tests contenant (p_values, success_rate, stats, critical_stats)
    hist = zeros((k, n_test, n_seuil, 4))

    # Mesurer le temps total d'exécution
    start_time = time.time()

    # Exécuter les tests
    for i in range(k):
        # Générer les données
        data = generator.generate(n)

        # Appliquer chaque test avec chaque seuil
        for t_idx, test in enumerate(tests):
            for a_idx, a in enumerate(alpha):
                # Exécuter le test
                p_val, stat_obs, stat_crit, success = test.test(data, a)

                # Stocker les résultats
                hist[i, t_idx, a_idx, 0] = p_val
                hist[i, t_idx, a_idx, 1] = stat_obs
                hist[i, t_idx, a_idx, 2] = stat_crit
                hist[i, t_idx, a_idx, 3] = int(success)  # bool -> int

    # Calculer le temps total d'exécution
    execution_time = time.time() - start_time

    # Calculer les statistiques agrégées
    results = {
        "mean_p_values": np.mean(hist[:, :, :, 0], axis=0),
        "mean_stats": np.mean(hist[:, :, :, 1], axis=0),
        "mean_critical_stats": np.mean(hist[:, :, :, 2], axis=0),
        "success_rate": np.mean(hist[:, :, :, 3], axis=0),
        "rejection_rate": 1 - np.mean(hist[:, :, :, 3], axis=0),
        "std_p_values": np.std(hist[:, :, :, 0], axis=0),
        "std_stats": np.std(hist[:, :, :, 1], axis=0),
        "min_p_values": np.min(hist[:, :, :, 0], axis=0),
        "max_p_values": np.max(hist[:, :, :, 0], axis=0),
        "statistic_gap": np.abs(np.mean(hist[:, :, :, 1], axis=0) - np.mean(hist[:, :, :, 2], axis=0)),
        "execution_time": execution_time,
        "total_iterations": k,
        "sample_size": n
    }

    # Afficher les résultats si demandé
    if display:
        print(f"Résultats des tests pour le générateur {generator.__class__.__name__}:")
        print(f"  Temps d'exécution total: {execution_time:.3f} secondes")
        print(f"  Nombre d'itérations: {k}")
        print(f"  Taille d'échantillon: {n}")
        print("\n  Taux de rejet par test et par seuil:")
        for t_idx, test in enumerate(tests):
            print(f"    {test.__class__.__name__}:")
            for a_idx, a in enumerate(alpha):
                print(f"      α={a}: {results['rejection_rate'][t_idx, a_idx]:.4f}")

    # Retourner les résultats selon le paramètre return_history
    if return_history:
        return results, hist
    else:
        return results
