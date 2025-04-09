def visualize_generator_comparison(gen1_results, gen2_results, test_names, alpha_values,
                                 metrics=None, plot_type="bar", x_axis="tests"):
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
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = ["mean_p_values", "mean_stats", "rejection_rate", "statistic_gap"]

    # Définir les titres et couleurs
    titles = {
        "mean_p_values": "P-valeurs moyennes",
        "mean_stats": "Statistiques observées moyennes",
        "rejection_rate": "Taux de rejet",
        "statistic_gap": "Écart statistique",
        "stability": "Stabilité des résultats",
        "execution_time": "Temps d'exécution"
    }

    colors = ["#3498db", "#e74c3c"]  # Bleu pour gen1, Rouge pour gen2

    # Créer les graphiques
    for metric in metrics:
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
