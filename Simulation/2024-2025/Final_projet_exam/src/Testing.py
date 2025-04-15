"""
    Module des fonctions pour appliquer automatiquement les tests sur les generateurs et/ou des séquences de nombres

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
"""
# import multiprocessing.pool

import time
import numpy as np
import  pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from numpy import ndarray , zeros
from functools import cache
from Tests import Tests , Chi2Test
from Generators import Generators
from typing import Dict, List, Callable, Tuple, Any
from dataclasses import dataclass
import os
from pathlib import Path


def test_generator(generator: Generators,
                  tests: List[Tests] = [Chi2Test()],
                  alpha: List[float] = [0.05],
                  k: int = 100,
                  n: int = 200000,
                  history: bool = False,
                  display: bool = False,
                  save: bool = False) -> Union[Dict, Tuple[Dict, pd.DataFrame]]:
    """
    Teste k fois un générateur avec plusieurs tests d'hypothèses et plusieurs seuils.
    Retourne une série de mesures telles que pour juger de la performance du générateur.

    Args:
        generator: Le générateur à tester.
        tests: Liste des tests d'hypothèses à appliquer.
        alpha: Liste des seuils utilisés pour les tests.
        k: Nombre d'itérations.
        n: Nombre d'éléments générés à chaque itération.
        return_history: Si True, retourne également l'historique complet des tests.
        display: Si True, affiche les résultats sous forme de tableaux.

    Returns:
        Si return_history est False: dictionnaire contenant les résultats agrégés.
        Si return_history est True: tuple (dictionnaire de résultats, historique complet sous forme de DataFrame).
    """
    # Calculer la taille totale du DataFrame
    total_rows = k * len(tests) * len(alpha)

    # Pré-allouer le DataFrame
    hist = pd.DataFrame({
        "Iteration": np.zeros(total_rows, dtype=int),
        "Test": [""] * total_rows,
        "Alpha": np.zeros(total_rows, dtype=float),
        "P-Value": np.zeros(total_rows, dtype=float),
        "Statistic": np.zeros(total_rows, dtype=float),
        "Critical Statistic": np.zeros(total_rows, dtype=float),
        "Success": np.zeros(total_rows, dtype=int)
    })

    # Mesurer le temps total d'exécution
    start_time = time.time()
    current_row = 0

    # Exécuter les tests
    for i in range(k):
        data = generator.generate(n)
        for test in tests:
            for a in alpha:
                res = test.test(data, a)

                # Remplir le DataFrame à l'index courant
                hist.iloc[current_row] = [
                    i,
                    str(test),
                    a,
                    res['p_value'],
                    res['stat_obs'],
                    res['stat_crit'],
                    int(res['accept'])
                ]
                current_row += 1

    if save:
        # Créer le répertoire data_results s'il n'existe pas
        save_dir = Path("data_results")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Construire le nom du fichier
        filename = f"test_{str(generator)}_k_{k}_n_{n}.csv"
        filepath = save_dir / filename

        # Sauvegarder le DataFrame
        hist.to_csv(filepath, sep=';', index=False)

    # Calculer le temps total d'exécution
    execution_time = time.time() - start_time

    # Calculer les statistiques agrégées
    results = {
        "mean_p_values": hist.groupby(["Test", "Alpha"])["P-Value"].mean(),
        "mean_stats": hist.groupby(["Test", "Alpha"])["Statistic"].mean(),
        "mean_critical_stats": hist.groupby(["Test", "Alpha"])["Critical Statistic"].mean(),
        "success_rate": hist.groupby(["Test", "Alpha"])["Success"].mean(),
        "rejection_rate": 1 - hist.groupby(["Test", "Alpha"])["Success"].mean(),
        "std_p_values": hist.groupby(["Test", "Alpha"])["P-Value"].std(),
        "std_stats": hist.groupby(["Test", "Alpha"])["Statistic"].std(),
        "min_p_values": hist.groupby(["Test", "Alpha"])["P-Value"].min(),
        "max_p_values": hist.groupby(["Test", "Alpha"])["P-Value"].max(),
        "statistic_gap": (hist.groupby(["Test", "Alpha"])["Statistic"].mean() -
                          hist.groupby(["Test", "Alpha"])["Critical Statistic"].mean()).abs(),
        "execution_time": execution_time,
        "total_iterations": k,
        "sample_size": n
    }

    # Afficher les résultats sous forme de tableaux si demandé
    if display:
        print(f"Résultats des tests pour le générateur {str(generator)}:")
        print(f"  Temps d'exécution total: {execution_time:.3f} secondes")
        print(f"  Nombre d'itérations: {k}")
        print(f"  Taille d'échantillon: {n}\n")
        max_car = max([len(str(metric)) for metric in results.keys()])
        # Afficher un tableau pour chaque métrique
        for metric, values in results.items():
            if isinstance(values, pd.Series):  # Afficher uniquement les métriques agrégées
                print(f"Tableau pour la métrique : {metric}")
                print("-"*(27+max_car))
                print(values.unstack(level=1))  # Réorganiser pour afficher les tests en colonnes
                print("-"*(27+max_car))
                print("\n")

    # Retourner les résultats selon le paramètre return_history
    if history:
        return results, hist
    else:
        return results

def test_sequence(sequence: np.ndarray,
                  tests: List[Tests] = [Chi2Test()],
                  alpha: List[float] = [0.05],
                  history :bool = False,
                  display: bool = False) -> Union[Dict, pd.DataFrame]:
    """
    Teste une séquence de nombres avec plusieurs tests d'hypothèses et plusieurs seuils.

    Args:
        sequence: La séquence de nombres à tester.
        tests: Liste des tests d'hypothèses à appliquer.
        alpha: Liste des seuils utilisés pour les tests.
        display: Si True, affiche les résultats sous forme de tableaux.

    Returns:
        Un dictionnaire contenant les résultats agrégés ou un DataFrame si display=True.
    """

    if len(sequence) == 0:
        raise ValueError("La séquence ne peut pas être vide")

    # Calculer la taille totale du DataFrame
    total_rows = len(tests) * len(alpha)

    # Pré-allouer le DataFrame pour stocker les résultats
    results_df = pd.DataFrame({
        "Test": [""] * total_rows,
        "Alpha": np.zeros(total_rows, dtype=float),
        "P-Value": np.zeros(total_rows, dtype=float),
        "Statistic": np.zeros(total_rows, dtype=float),
        "Critical Statistic": np.zeros(total_rows, dtype=float),
        "Success": np.zeros(total_rows, dtype=int)
    })

    # Index pour remplir le DataFrame
    current_row = 0

    # Appliquer chaque test avec chaque seuil
    for t in tests:
        for a in alpha:
            # Exécuter le test
            res = t.test(sequence, a)

            # Remplir le DataFrame à l'index courant
            results_df.iloc[current_row] = [
                str(t),
                a,
                res['p_value'],
                res['stat_obs'],
                res['stat_crit'],
                int(res['accept'])
            ]
            current_row += 1

    # Calculer les statistiques agrégées
    aggregated_results = {
        "mean_p_values": results_df.groupby(["Test", "Alpha"])["P-Value"].mean(),
        "mean_stats": results_df.groupby(["Test", "Alpha"])["Statistic"].mean(),
        "mean_critical_stats": results_df.groupby(["Test", "Alpha"])["Critical Statistic"].mean(),
        "success_rate": results_df.groupby(["Test", "Alpha"])["Success"].mean(),
        "rejection_rate": 1 - results_df.groupby(["Test", "Alpha"])["Success"].mean(),
        "std_p_values": results_df.groupby(["Test", "Alpha"])["P-Value"].std(),
        "std_stats": results_df.groupby(["Test", "Alpha"])["Statistic"].std(),
        "min_p_values": results_df.groupby(["Test", "Alpha"])["P-Value"].min(),
        "max_p_values": results_df.groupby(["Test", "Alpha"])["P-Value"].max(),
        "statistic_gap": (results_df.groupby(["Test", "Alpha"])["Statistic"].mean() -
                          results_df.groupby(["Test", "Alpha"])["Critical Statistic"].mean()).abs()
    }

    # Afficher les résultats sous forme de tableaux si demandé
    if display:
        print("Résultats des tests pour la séquence :")
        print(f"  Longueur de la séquence: {len(sequence)}\n")

        # Afficher un tableau pour chaque métrique
        for metric, values in aggregated_results.items():
            if isinstance(values, pd.Series):  # Afficher uniquement les métriques agrégées
                print(f"Tableau pour la métrique : {metric}")
                print(values.unstack(level=0))  # Réorganiser pour afficher les tests en colonnes
                print("\n")

    if history :
        return aggregated_results , results_df
    else :
        return aggregated_results



@dataclass
class ScoreMetric:
    """
        classe implementant le score par rapport à une métrique
    """
    name: str # nom de la métrique
    function: Callable # evaluation du score par rapport à la métrique
    lower_is_better: bool = True # c'est celui qui a le plus petit score qui gagne
    weight: float = 1.0 # point ajouté au winner

def enhanced_score(
    gen1: Generators,
    gen2: Generators,
    metrics: List[ScoreMetric],
    info: bool = False
) -> Dict:
    """
    Compare deux générateurs avec un système de score pondéré et détaillé.

    Args:
        gen1: Premier générateur
        gen2: Second générateur
        metrics: Liste des métriques à utiliser
        info: Si True, retourne les détails

    Returns:
        Tuple des scores ou dictionnaire détaillé
    """
    scores = {"gen1": 0.0, "gen2": 0.0}
    details = {}
    total_points = 0
    for metric in metrics:
        total_points=+1
        val1 = metric.function(gen1)
        val2 = metric.function(gen2)

        # Déterminer le gagnant
        is_gen1_better = (val1 < val2) if metric.lower_is_better else (val1 > val2)

        if is_gen1_better:
            scores["gen1"] += metric.weight
            winner = "gen1"
        else:
            scores["gen2"] += metric.weight
            winner = "gen2"

        details[metric.name] = {
            "": val1,
            "gen2_value": val2,
            "winner": winner,
            "difference": (val1 , val2) ,
        }

    if info:
        return {
            "scores": scores,
            "details": details,
            "total points": total_points
        }

    return scores

def analyze_sequence_structure(sequence: np.ndarray,
                            window_size: int = 1000,
                            overlap: int = 500) -> Dict:
    """
    Analyse la structure locale d'une séquence.
    """
    results = {
        'local_uniformity': [], # pour verifier la localité uniforme
        'serial_correlation': [], # pour verifier les
        'runs_score': []
    }

    # Analyse par fenêtre glissante
    for i in range(0, len(sequence) - window_size, overlap):
        window = sequence[i:i + window_size]

        # Tests locaux
        results['local_uniformity'].append(stats.kstest(window, 'uniform').pvalue)
        results['serial_correlation'].append(np.corrcoef(window[:-1], window[1:])[0,1])

    return {
        'mean_local_uniformity': np.mean(results['local_uniformity']),
        'mean_serial_correlation': np.mean(results['serial_correlation']),
        'stability': np.std(results['local_uniformity']),
        'detailed_results': results
    }


def analyze_convergence(generator: Generators,
                       tests: List[Tests],
                       alpha: float = 0.05,
                       n_range: List[int] = [1000, 5000, 10000, 50000, 100000],
                       k: int = 100) -> Dict:
    """
    Analyse la convergence des tests pour un générateur donné.
    """
    convergence_data = {
        'n_values': [],
        'p_values': [],
        'rejection_rates': [],
        'stability_scores': []
    }

    for n in n_range:
        results = test_generator(generator, tests=tests, alpha=[alpha], k=k, n=n)

        # Calculer les métriques de convergence
        rejection_rate = results['rejection_rate'].mean()
        p_value_stability = 1 - results['std_p_values'].mean()

        convergence_data['n_values'].append(n)
        convergence_data['rejection_rates'].append(rejection_rate)
        convergence_data['stability_scores'].append(p_value_stability)

    # Calculer les scores de convergence
    conv_score = np.mean(np.diff(convergence_data['rejection_rates']))
    stab_score = np.mean(convergence_data['stability_scores'])

    return {
        'convergence_score': conv_score,
        'stability_score': stab_score,
        'data': convergence_data
    }

def compare_generators_score(gen1: Generators,
                           gen2: Generators,
                           tests: List[Tests],
                           alpha: float = 0.05,
                           n: int = 10000,
                           k: int = 100) -> Dict:
    """
    Compare deux générateurs sur différentes métriques et attribue des scores.
    """
    # Obtenir les résultats pour chaque générateur
    results1 = test_generator(gen1, tests=tests, alpha=[alpha], k=k, n=n)
    results2 = test_generator(gen2, tests=tests, alpha=[alpha], k=k, n=n)

    # Initialiser les scores
    scores = {
        'gen1': 0,
        'gen2': 0,
        'metrics': {}
    }

    # Métriques à comparer
    metrics = {
        'rejection_rate': lambda x: abs(x - alpha),  # Plus proche de alpha est meilleur
        'std_p_values': lambda x: x,                 # Plus petit est meilleur
        'execution_time': lambda x: x,               # Plus petit est meilleur
        'statistic_gap': lambda x: x                 # Plus petit est meilleur
    }

    # Comparer chaque métrique
    for metric, scorer in metrics.items():
        val1 = scorer(results1[metric].mean())
        val2 = scorer(results2[metric].mean())

        # Attribuer le point au meilleur
        if val1 < val2:
            scores['gen1'] += 1
            winner = 'gen1'
        elif val2 < val1:
            scores['gen2'] += 1
            winner = 'gen2'
        else:
            winner = 'tie'

        scores['metrics'][metric] = {
            'gen1': val1,
            'gen2': val2,
            'winner': winner
        }

    return scores