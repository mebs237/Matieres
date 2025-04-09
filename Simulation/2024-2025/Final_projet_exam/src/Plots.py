import matplotlib.pyplot as plt
import numpy as np
from Testing import test_generator
import pandas as pd

def compare_generators(gen1, gen2, test_names, alpha):
    """
    Compare visuellement deux générateurs à partir des résultats de test_generator.

    Args:
        gen1_results: Résultats du générateur 1 (dict contenant 'mean_p_values', etc.)
        gen2_results: Résultats du générateur 2 (même format).
        test_names: Liste des noms des tests.
        alpha: Liste des niveaux de significativité utilisés.
    """

    measures = ['mean_p_values', 'mean_stats', 'mean_critical_stats', 'success_rate']
    titles = {
        'mean_p_values': "P-valeurs moyennes",
        'mean_stats': "Statistiques observées moyennes",
        'mean_critical_stats': "Statistiques critiques moyennes",
        'success_rate': "Taux de réussite des tests"
    }

    n_tests = len(test_names)
    n_alpha = len(alpha)
    gen1_results= test_generator(gen1,tests=test_names,alpha=alpha)
    gen2_results= test_generator(gen2,tests=test_names,alpha=alpha)
    bar_width = 0.35

    x = np.arange(n_tests)  # Position de base des tests

    for m in measures:
        for a_idx, a in enumerate(alpha):
            plt.figure(figsize=(10, 5))

            gen1 = gen1_results[m][:, a_idx]
            gen2 = gen2_results[m][:, a_idx]

            # Position des barres
            x_gen1 = x - bar_width / 2
            x_gen2 = x + bar_width / 2

            plt.bar(x_gen1, gen1, width=bar_width, color='blue', label='Générateur 1')
            plt.bar(x_gen2, gen2, width=bar_width, color='red', label='Générateur 2')

            plt.xticks(x, test_names, rotation=45)
            plt.ylabel(titles[m])
            plt.title(f"{titles[m]} (alpha = {a})")
            plt.legend()
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.show()

def compare_generators_metrics(gen1, gen2, test_names, alpha):
    """
    Compare deux générateurs en utilisant plusieurs métriques statistiques.

    Args:
        gen1_results: Résultats du générateur 1
        gen2_results: Résultats du générateur 2
        test_names: Liste des noms des tests
        alpha: Liste des niveaux de significativité

    Returns:
        DataFrame contenant les comparaisons pour chaque test et chaque alpha
    """
    gen1_results = test_generator(gen1,tests=test_names,alpha=alpha)
    gen2_results = test_generator(gen2,tests=test_names,alpha=alpha)
    # Créer un DataFrame pour stocker les résultats
    comparison = []

    for t_idx, test_name in enumerate(test_names):
        for a_idx, a in enumerate(alpha):
            # Calculer les différences relatives
            p_val_diff = (gen1_results["mean_p_values"][t_idx, a_idx] -
                         gen2_results["mean_p_values"][t_idx, a_idx]) / gen2_results["mean_p_values"][t_idx, a_idx]

            stat_diff = (gen1_results["mean_stats"][t_idx, a_idx] -
                        gen2_results["mean_stats"][t_idx, a_idx]) / gen2_results["mean_stats"][t_idx, a_idx]

            rejection_diff = (gen1_results["rejection_rate"][t_idx, a_idx] -
                            gen2_results["rejection_rate"][t_idx, a_idx])

            # Déterminer quel générateur est meilleur pour chaque métrique
            better_p_val = "Générateur 1" if gen1_results["mean_p_values"][t_idx, a_idx] > gen2_results["mean_p_values"][t_idx, a_idx] else "Générateur 2"
            better_stat = "Générateur 1" if abs(gen1_results["mean_stats"][t_idx, a_idx] - gen1_results["mean_critical_stats"][t_idx, a_idx]) < abs(gen2_results["mean_stats"][t_idx, a_idx] - gen2_results["mean_critical_stats"][t_idx, a_idx]) else "Générateur 2"
            better_rejection = "Générateur 1" if gen1_results["rejection_rate"][t_idx, a_idx] < gen2_results["rejection_rate"][t_idx, a_idx] else "Générateur 2"

            comparison.append({
                "Test": test_name,
                "Alpha": a,
                "G1_P_Value": gen1_results["mean_p_values"][t_idx, a_idx],
                "G2_P_Value": gen2_results["mean_p_values"][t_idx, a_idx],
                "P_Value_Diff_%": p_val_diff * 100,
                "G1_Stat": gen1_results["mean_stats"][t_idx, a_idx],
                "G2_Stat": gen2_results["mean_stats"][t_idx, a_idx],
                "Stat_Diff_%": stat_diff * 100,
                "G1_Rejection_Rate": gen1_results["rejection_rate"][t_idx, a_idx],
                "G2_Rejection_Rate": gen2_results["rejection_rate"][t_idx, a_idx],
                "Rejection_Diff": rejection_diff,
                "Better_P_Value": better_p_val,
                "Better_Stat": better_stat,
                "Better_Rejection": better_rejection
            })

    return pd.DataFrame(comparison)
