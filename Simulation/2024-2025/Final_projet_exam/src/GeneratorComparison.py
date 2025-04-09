"""
Module pour la comparaison et la visualisation des générateurs aléatoires.
Fournit des fonctions pour calculer des scores et visualiser les résultats.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Callable
from Testing import test_generator
from Generators import Generator
from Tests import Test

def calculate_generator_score(gen1_value:float,
                              gen2_value:float,                       metric_type: str,
                              alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calcule un score pour deux générateurs selon une métrique donnée.

    Args:
        gen1_value: Valeur de la métrique pour le générateur 1
        gen2_value: Valeur de la métrique pour le générateur 2
        metric_type: Type de métrique ('rejection_rate', 'p_value', 'statistic_gap', 'stability', 'execution_time')
        alpha: Niveau de signification pour les tests (défaut: 0.05)

    Returns:
        Tuple (score_gen1, score_gen2) où chaque score est entre 0 et 1
    """
    if metric_type == "rejection_rate":
        # Pour le taux de rejet, le meilleur est celui le plus proche de alpha
        gen1_score = 1 - abs(gen1_value - alpha)
        gen2_score = 1 - abs(gen2_value - alpha)
    elif metric_type == "mean_p_value":
        # Pour les p-valeurs, le meilleur est celui le plus proche de 0.5
        gen1_score = 1 - abs(gen1_value - 0.5) * 2
        gen2_score = 1 - abs(gen2_value - 0.5) * 2
    elif metric_type == "statistic_gap":
        # Pour l'écart statistique, le meilleur est celui avec le plus petit écart
        gen1_score = 1 / (1 + gen1_value)
        gen2_score = 1 / (1 + gen2_value)
    elif metric_type == "std_stats":
        # Pour la stabilité, le meilleur est celui avec la plus petite variance
        gen1_score = 1 / (1 + gen1_value)
        gen2_score = 1 / (1 + gen2_value)
    elif metric_type == "std_p_values":
        # Pour la stabilité, le meilleur est celui avec la plus petite variance
        gen1_score = 1 / (1 + gen1_value)
        gen2_score = 1 / (1 + gen2_value)
    elif metric_type == "execution_time":
        # Pour le temps d'exécution, le meilleur est le plus rapide
        gen1_score = 1 / gen1_value
        gen2_score = 1 / gen2_value
    else:
        # Par défaut, on suppose que la plus grande valeur est la meilleure
        gen1_score = gen1_value
        gen2_score = gen2_value

    # Normaliser les scores pour qu'ils somment à 1
    total = gen1_score + gen2_score
    if total > 0:
        gen1_score = gen1_score / total
        gen2_score = gen2_score / total
    else:
        gen1_score = 0.5
        gen2_score = 0.5

    return gen1_score, gen2_score

def compare_generators(gen1: Generator, gen2: Generator,
                      tests: List[Test], alpha: List[float] = [0.05],
                      k: int = 100, n: int = 200000,
                      metrics: Optional[List[str]] = None) -> Dict:
    """
    Compare deux générateurs en utilisant plusieurs métriques statistiques.

    Args:
        gen1: Premier générateur à comparer
        gen2: Second générateur à comparer
        tests: Liste des tests à appliquer
        alpha: Liste des niveaux de signification
        k: Nombre d'itérations
        n: Nombre d'éléments générés à chaque itération
        metrics: Liste des métriques à calculer (par défaut: toutes)

    Returns:
        Dictionnaire contenant les comparaisons pour chaque test et chaque alpha
    """
    # Obtenir les résultats des tests
    gen1_results = test_generator(gen1, alpha=alpha, k=k, n=n, tests=tests)
    gen2_results = test_generator(gen2, alpha=alpha, k=k, n=n, tests=tests)

    # Définir les métriques à calculer
    if metrics is None:
        metrics = ["rejection_rate", "mean_p_values", "statistic_gap", "execution_time"]

    # Créer un dictionnaire pour stocker les résultats
    comparison = {}

    # Calculer les scores pour chaque test, chaque alpha et chaque métrique
    for t_idx, test in enumerate(tests):
        test_name = str(test)
        comparison[test_name] = {}

        for a_idx, a in enumerate(alpha):
            alpha_str = f"alpha_{a:.2f}"
            comparison[test_name][alpha_str] = {}

            for metric in metrics:
                if metric in gen1_results and metric in gen2_results:
                    gen1_value = gen1_results[metric][t_idx, a_idx]
                    gen2_value = gen2_results[metric][t_idx, a_idx]

                    gen1_score, gen2_score = calculate_generator_score(
                        gen1_value, gen2_value, metric, a)

                    comparison[test_name][alpha_str][metric] = {
                        "gen1_value": gen1_value,
                        "gen2_value": gen2_value,
                        "gen1_score": gen1_score,
                        "gen2_score": gen2_score,
                        "better_generator": "Générateur 1" if gen1_score > gen2_score else "Générateur 2"
                    }

    # Calculer un score global pour chaque générateur
    gen1_total_score = 0
    gen2_total_score = 0
    total_comparisons = 0

    for test_name, test_data in comparison.items():
        for alpha_str, alpha_data in test_data.items():
            for metric, metric_data in alpha_data.items():
                gen1_total_score += metric_data["gen1_score"]
                gen2_total_score += metric_data["gen2_score"]
                total_comparisons += 1

    if total_comparisons > 0:
        gen1_total_score /= total_comparisons
        gen2_total_score /= total_comparisons

    comparison["global_score"] = {
        "gen1_score": gen1_total_score,
        "gen2_score": gen2_total_score,
        "better_generator": "Générateur 1" if gen1_total_score > gen2_total_score else "Générateur 2"
    }

    return comparison

def visualize_generator_comparison(gen1_results: Dict, gen2_results: Dict,
                                 test_names: List[str], alpha_values: List[float],
                                 metrics: Optional[List[str]] = None,
                                 plot_type: str = "bar",
                                 x_axis: str = "tests") -> None:
    """
    Visualise la comparaison entre deux générateurs.

    Args:
        gen1_results: Résultats du premier générateur
        gen2_results: Résultats du second générateur
        test_names: Liste des noms des tests
        alpha_values: Liste des valeurs alpha utilisées
        metrics: Liste des métriques à visualiser (par défaut: toutes)
        plot_type: Type de graphique ("bar" ou "line")
        x_axis: Variable en abscisse ("tests", "alpha", "n", "k")
    """
    if metrics is None:
        metrics = ["mean_p_values", "mean_stats", "rejection_rate", "statistic_gap"]

    # Définir les titres et couleurs
    titles = {
        "mean_p_values": "P-valeurs moyennes",
        "mean_stats": "Statistiques observées moyennes",
        "mean_critical_stats": "Statistiques critiques moyennes",
        "success_rate": "Taux de réussite des tests",
        "rejection_rate": "Taux de rejet",
        "statistic_gap": "Écart statistique",
        "std_p_values": "Écart-type des p-valeurs",
        "std_stats": "Écart-type des statistiques",
        "execution_time": "Temps d'exécution"
    }

    colors = ["#3498db", "#e74c3c"]  # Bleu pour gen1, Rouge pour gen2

    # Créer les graphiques
    for metric in metrics:
        if metric not in gen1_results or metric not in gen2_results:
            print(f"Métrique '{metric}' non disponible dans les résultats")
            continue

        plt.figure(figsize=(12, 6))

        if plot_type == "bar":
            if x_axis == "tests":
                x = np.arange(len(test_names))
                width = 0.35

                for i, alpha in enumerate(alpha_values):
                    plt.bar(x - width/2, gen1_results[metric][:, i], width,
                           label=f"Générateur 1 (α={alpha})", color=colors[0], alpha=0.7)
                    plt.bar(x + width/2, gen2_results[metric][:, i], width,
                           label=f"Générateur 2 (α={alpha})", color=colors[1], alpha=0.7)

                plt.xlabel("Tests")
                plt.xticks(x, test_names, rotation=45)

            elif x_axis == "alpha":
                x = alpha_values

                for i, test in enumerate(test_names):
                    plt.plot(x, gen1_results[metric][i, :], 'o-',
                            label=f"Générateur 1 - {test}", color=colors[0])
                    plt.plot(x, gen2_results[metric][i, :], 'o-',
                            label=f"Générateur 2 - {test}", color=colors[1])

                plt.xlabel("Niveau de signification (α)")

        plt.ylabel(titles.get(metric, metric))
        plt.title(f"Comparaison des {titles.get(metric, metric)}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_generator_evolution(gen1: Generator, gen2: Generator,
                           tests: List, alpha: float = 0.05,
                           k: int = 100, n_values: List[int] = [10000, 50000, 100000, 200000],
                           metric: str = "rejection_rate") -> None:
    """
    Trace l'évolution d'une métrique en fonction de la taille de l'échantillon.

    Args:
        gen1: Premier générateur
        gen2: Second générateur
        tests: Liste des tests à appliquer
        alpha: Niveau de signification
        k: Nombre d'itérations
        n_values: Liste des tailles d'échantillon à tester
        metric: Métrique à visualiser
    """
    n_tests = len(tests)
    n_sizes = len(n_values)

    # Initialiser les tableaux de résultats
    gen1_results = np.zeros((n_tests, n_sizes))
    gen2_results = np.zeros((n_tests, n_sizes))

    # Calculer les résultats pour chaque taille d'échantillon
    for i, n in enumerate(n_values):
        gen1_data = test_generator(gen1, alpha=[alpha], k=k, n=n, tests=tests)
        gen2_data = test_generator(gen2, alpha=[alpha], k=k, n=n, tests=tests)

        for t_idx in range(n_tests):
            gen1_results[t_idx, i] = gen1_data[metric][t_idx, 0]
            gen2_results[t_idx, i] = gen2_data[metric][t_idx, 0]

    # Tracer les résultats
    plt.figure(figsize=(12, 6))

    for t_idx, test in enumerate(tests):
        plt.plot(n_values, gen1_results[t_idx, :], 'o-',
                label=f"Générateur 1 - {test.__class__.__name__}", color="#3498db")
        plt.plot(n_values, gen2_results[t_idx, :], 'o-',
                label=f"Générateur 2 - {test.__class__.__name__}", color="#e74c3c")

    plt.xlabel("Taille de l'échantillon (n)")
    plt.ylabel(metric)
    plt.title(f"Évolution de {metric} en fonction de la taille de l'échantillon")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_generator_iterations(gen1: Generator, gen2: Generator,
                            tests: List, alpha: float = 0.05,
                            n: int = 200000, k_values: List[int] = [10, 50, 100, 200],
                            metric: str = "rejection_rate") -> None:
    """
    Trace l'évolution d'une métrique en fonction du nombre d'itérations.

    Args:
        gen1: Premier générateur
        gen2: Second générateur
        tests: Liste des tests à appliquer
        alpha: Niveau de signification
        n: Taille de l'échantillon
        k_values: Liste des nombres d'itérations à tester
        metric: Métrique à visualiser
    """
    n_tests = len(tests)
    n_iterations = len(k_values)

    # Initialiser les tableaux de résultats
    gen1_results = np.zeros((n_tests, n_iterations))
    gen2_results = np.zeros((n_tests, n_iterations))

    # Calculer les résultats pour chaque nombre d'itérations
    for i, k in enumerate(k_values):
        gen1_data = test_generator(gen1, alpha=[alpha], k=k, n=n, tests=tests)
        gen2_data = test_generator(gen2, alpha=[alpha], k=k, n=n, tests=tests)

        for t_idx in range(n_tests):
            gen1_results[t_idx, i] = gen1_data[metric][t_idx, 0]
            gen2_results[t_idx, i] = gen2_data[metric][t_idx, 0]

    # Tracer les résultats
    plt.figure(figsize=(12, 6))

    for t_idx, test in enumerate(tests):
        plt.plot(k_values, gen1_results[t_idx, :], 'o-',
                label=f"Générateur 1 - {test.__class__.__name__}", color="#3498db")
        plt.plot(k_values, gen2_results[t_idx, :], 'o-',
                label=f"Générateur 2 - {test.__class__.__name__}", color="#e74c3c")

    plt.xlabel("Nombre d'itérations (k)")
    plt.ylabel(metric)
    plt.title(f"Évolution de {metric} en fonction du nombre d'itérations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()