"""
Module amélioré pour l'analyse automatique des séquences et l'évaluation des générateurs
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from Tests import Tests, Chi2Test, KSTest
from Generators import Generators
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScoreMetric:
    """
    Classe améliorée implémentant une métrique des tests.
    """
    name: str
    func: Callable[[dict], float]
    lower_is_better: bool = True
    weight: float = 1.0
    description: str = ""

    def compute(self, result: dict) -> float:
        """Calcule le score en appliquant la fonction de métrique au résultat"""
        try:
            return self.weight * self.func(result)
        except (KeyError, TypeError) as e:
            logger.warning(f"Erreur dans le calcul de la métrique {self.name}: {e}")
            return 0.0

DEFAULT_METRICS = [
    ScoreMetric(
        name="convergence",
        func=lambda x: x.get("convergence_score", 0),
        weight=2.0,
        description="Score de convergence des p-values"
    ),
    ScoreMetric(
        name="stability",
        func=lambda x: x.get("stability_score", 0),
        weight=1.5,
        description="Stabilité des résultats sur différentes tailles de fenêtres"
    ),
    ScoreMetric(
        name="mean_p_value",
        func=lambda r: r.get("stats", {}).get("mean_p_value", 0),
        lower_is_better=False,
        weight=1.5,
        description="Moyenne des p-values"
    ),
    ScoreMetric(
        name="accept_ratio",
        func=lambda r: r.get("stats", {}).get("accept_ratio", 0),
        lower_is_better=False,
        weight=2.0,
        description="Taux d'acceptation de l'hypothèse nulle"
    ),
    ScoreMetric(
        name="violation_ratio",
        func=lambda r: r.get("stats", {}).get("violation_ratio", 0),
        lower_is_better=True,
        weight=1.0,
        description="Taux de violation des seuils critiques"
    ),
    ScoreMetric(
        name="std_p_value",
        func=lambda r: r.get("stats", {}).get("std_p_value", 0),
        lower_is_better=True,
        weight=0.5,
        description="Écart-type des p-values"
    )
]

@dataclass
class AnalysisResult:
    """
    Classe améliorée pour l'encapsulation des résultats d'analyse.
    """
    hist_df: pd.DataFrame
    stats: dict
    name: str = field(default_factory=lambda: f"Result_{AnalysisResult._increment}")
    window_sizes: List[int] = field(default_factory=list)
    _increment: int = 0

    def __post_init__(self):
        AnalysisResult._increment += 1

    def __str__(self) -> str:
        return f"Analysis {self.name} with {len(self.hist_df)} points"

    def generate_report(self) -> str:
        """Génère un rapport d'interprétation amélioré."""
        report = f"# Rapport d'interprétation de {self.name}\n\n"
        report += "## Résumé global\n"
        report += f"- Taille de la séquence analysée: {len(self.hist_df)}\n"
        report += f"- Nombre de fenêtres analysées: {len(self.hist_df['window_size'].unique())}\n"

        for test_name, stats in self.stats.items():
            report += f"\n## Résultats pour {test_name}\n"
            report += f"```json\n{json.dumps({k: round(v, 4) for k, v in stats.items()}, indent=2)}\n```\n"

            # Interprétation détaillée
            report += self._interpret_test_results(stats)

        return report

    def _interpret_test_results(self, stats: dict) -> str:
        """Génère une interprétation détaillée des résultats."""
        interpretation = "### Interprétation:\n"

        # Analyse de la p-value moyenne
        mean_p = stats.get('mean_p_value', 0.5)
        if mean_p > 0.7:
            interpretation += f"- Excellente conformité (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.5:
            interpretation += f"- Bonne conformité (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.3:
            interpretation += f"- Conformité modérée (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.1:
            interpretation += f"- Conformité faible (p-value moyenne: {mean_p:.3f})\n"
        else:
            interpretation += f"- Non-conformité (p-value moyenne: {mean_p:.3f})\n"

        # Analyse du taux d'acceptation
        acc_ratio = stats.get('accept_ratio', 0)
        alpha = stats.get('alpha', 0.05)
        expected_ratio = 1 - alpha

        if acc_ratio > expected_ratio * 1.1:
            interpretation += f"- Taux d'acceptation anormalement élevé ({acc_ratio:.1%} vs {expected_ratio:.1%} attendu)\n"
        elif acc_ratio > expected_ratio * 0.9:
            interpretation += f"- Taux d'acceptation normal ({acc_ratio:.1%})\n"
        else:
            interpretation += f"- Taux d'acceptation trop bas ({acc_ratio:.1%} vs {expected_ratio:.1%} attendu)\n"

        return interpretation

    def save_all(self, output_dir: Union[str, Path] = "results") -> None:
        """Sauvegarde tous les résultats dans un dossier."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self.save_report(output_path)
        self.save_hist(output_path)
        self.save_fig(output_path)

    def save_report(self, output_path: Path) -> None:
        """Sauvegarde le rapport."""
        report_path = output_path / f"report_{self.name}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.generate_report())
        logger.info("Rapport sauvegardé dans %s", report_path)

    def save_hist(self, output_path: Path) -> None:
        """Sauvegarde l'historique des résultats."""
        hist_path = output_path / f"history_{self.name}.csv"
        self.hist_df.to_csv(hist_path, index=False)
        logger.info("Historique sauvegardé dans %s", hist_path)

    def save_fig(self, output_path: Path) -> None:
        """Sauvegarde les visualisations."""
        try:
            fig_path = output_path / f"plots_{self.name}.html"

            # Crée un dashboard avec plusieurs graphiques
            fig = go.Figure()

            # Graphique des p-values par fenêtre
            fig.add_trace(go.Scatter(
                x=self.hist_df["window_start"],
                y=self.hist_df["p_value"],
                mode="lines+markers",
                name="p-values"
            ))

            # Ajoute une ligne pour le seuil alpha
            alpha = self.hist_df["alpha"].mean()
            fig.add_hline(y=alpha, line_dash="dash", line_color="red",
                         annotation_text=f"Seuil α={alpha}",
                         annotation_position="bottom right")

            fig.update_layout(
                title=f"Évolution des p-values - {self.name}",
                xaxis_title="Position de la fenêtre",
                yaxis_title="p-value"
            )

            fig.write_html(fig_path)
            logger.info("Graphiques sauvegardés dans %s", fig_path)

        except (OSError, ValueError) as e:
            logger.error("Erreur lors de la sauvegarde des graphiques: %s", e)

class EvaluationResult(AnalysisResult):
    """
    Classe améliorée pour l'évaluation des générateurs.
    """
    def __init__(self, hist_df: pd.DataFrame, stats: dict,
                 generator_name: str = None, window_sizes: List[int] = None,
                 metrics: List[ScoreMetric] = None, **kwargs):
        super().__init__(hist_df, stats, window_sizes=window_sizes, **kwargs)
        self.generator_name = generator_name
        self.metrics = metrics or DEFAULT_METRICS

    def generate_fig(self, plot_type: str = "dashboard") -> go.Figure:
        """Génère des visualisations améliorées."""
        if plot_type == "dashboard":
            return self._generate_dashboard()
        elif plot_type == "p_value":
            return self._generate_p_value_plot()
        elif plot_type == "distribution":
            return self._generate_distribution_plot()
        else:
            raise ValueError(f"Type de graphique non supporté: {plot_type}")

    def _generate_dashboard(self) -> go.Figure:
        """Génère un dashboard complet."""
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("Évolution des p-values",
                                          "Distribution des p-values",
                                          "Taux d'acceptation par fenêtre",
                                          "Statistiques globales"))

        # Graphique 1: Évolution des p-values
        for test_name, group in self.hist_df.groupby("test_name"):
            fig.add_trace(
                go.Scatter(x=group["window_start"], y=group["p_value"],
                          name=test_name, mode="lines+markers"),
                row=1, col=1
            )

        # Graphique 2: Distribution des p-values
        fig.add_trace(
            go.Histogram(x=self.hist_df["p_value"], nbinsx=20,
                        name="Distribution p-values"),
            row=1, col=2
        )

        # Graphique 3: Taux d'acceptation par fenêtre
        acceptance = self.hist_df.groupby("window_size")["accept"].mean().reset_index()
        fig.add_trace(
            go.Bar(x=acceptance["window_size"], y=acceptance["accept"],
                  name="Taux d'acceptation"),
            row=2, col=1
        )

        # Graphique 4: Statistiques globales
        metrics = ["mean_p_value", "accept_ratio", "violation_ratio"]
        fig.add_trace(
            go.Bar(x=metrics,
                  y=[self.stats.get(m, 0) for m in metrics],
                  name="Métriques"),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1000,
                         title_text=f"Analyse du générateur {self.generator_name}")
        return fig

def _process_window(data: NDArray, tests: List[Tests], alpha: float,
                   window_start: int, window_size: int) -> List[dict]:
    """Traite une fenêtre de données (fonction helper pour le traitement parallèle)."""
    results = []
    sub_seq = data[window_start:window_start + window_size]

    for test in tests:
        try:
            res = test.test(sub_seq, alpha=alpha)
            res["test_name"] = str(test)
            res["window_start"] = window_start
            res["window_size"] = window_size
            results.append(res)
        except Exception as e:
            logger.error(f"Erreur dans le test {test} sur la fenêtre {window_start}:{window_start+window_size}: {e}")

    return results

def analyse_sequence(sequence: NDArray,
                     tests: Optional[List[Tests]] = None,
                     alpha: float = 0.05,
                     granularities: Union[int, float, List[Union[int, float]]] = 0.1,
                     parallel: bool = True) -> AnalysisResult:
    """
    Version améliorée de la fonction d'analyse de séquence.

    Ajouts:
    - Traitement parallèle des fenêtres
    - Meilleure gestion des erreurs
    - Optimisation des calculs
    """
    tests = tests or [Chi2Test(), KSTest()]
    n = len(sequence)

    if not isinstance(granularities, list):
        granularities = [granularities]

    window_sizes = []
    for g in granularities:
        if isinstance(g, float) and 0 < g <= 1:
            window_sizes.append(int(g * n))
        elif isinstance(g, int) and 0 < g <= n:
            window_sizes.append(g)
        else:
            raise ValueError(f"Granularité invalide: {g}. Doit être un entier <= {n} ou un float <= 1")

    # Traitement parallèle des fenêtres
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for wind_size in window_sizes:
            for i in range(0, n - wind_size + 1, wind_size):
                futures.append(
                    executor.submit(
                        _process_window,
                        data=sequence,
                        tests=tests,
                        alpha=alpha,
                        window_start=i,
                        window_size=wind_size
                    )
                )

        for future in futures:
            results.extend(future.result())

    if not results:
        raise ValueError("Aucun résultat n'a été généré - vérifiez les paramètres")

    hist_df = pd.DataFrame(results)

    # Calcul des statistiques globales
    stats = {
        "mean_p_value": hist_df["p_value"].mean(),
        "median_p_value": hist_df["p_value"].median(),
        "accept_ratio": hist_df["accept"].mean(),
        "rejection_ratio": 1 - hist_df["accept"].mean(),
        "violation_ratio": hist_df["violation"].mean(),
        "std_p_value": hist_df["p_value"].std(),
        "alpha": alpha
    }

    return AnalysisResult(hist_df=hist_df, stats=stats, window_sizes=window_sizes)

def evaluate_generator(generator: Generators,
                      tests: Optional[List[Tests]] = None,
                      granularities: List[Union[int, float]] = None,
                      alpha: float = 0.05,
                      n_repeat: int = 200,
                      seq_len: int = 10_000,
                      parallel: bool = True) -> EvaluationResult:
    """
    Version améliorée de la fonction d'évaluation des générateurs.
    """
    tests = tests or [Chi2Test(), KSTest()]
    granularities = granularities or [0.1]

    # Utilisation d'une liste pour collecter les résultats de manière thread-safe
    all_results = []

    def process_iteration(i: int):
        try:
            seq = generator.generate(seq_len)
            res = analyse_sequence(
                sequence=seq,
                alpha=alpha,
                tests=tests,
                granularities=granularities,
                parallel=parallel
            )
            return {**res.stats, "iteration": i}
        except Exception as e:
            logger.error(f"Erreur dans l'itération {i}: {e}")
            return None

    # Exécution parallèle des itérations
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_iteration, i) for i in range(n_repeat)]
        for future in futures:
            result = future.result()
            if result is not None:
                all_results.append(result)

    if not all_results:
        raise ValueError("Aucun résultat valide n'a été généré")

    df = pd.DataFrame(all_results)

    stats = {
        "mean_p_value": df["mean_p_value"].mean(),
        "median_p_value": df["mean_p_value"].median(),
        "accept_ratio": df["accept_ratio"].mean(),
        "rejection_ratio": df["rejection_ratio"].mean(),
        "violation_ratio": df["violation_ratio"].mean(),
        "std_p_value": df["mean_p_value"].std(),
        "alpha": alpha,
        "n_iterations": n_repeat,
        "sequence_length": seq_len
    }

    return EvaluationResult(
        hist_df=df,
        stats=stats,
        generator_name=str(generator),
        window_sizes=[g if isinstance(g, int) else int(g*seq_len) for g in granularities]
    )

def compare_generators(gen1: Generators,
                      gen2: Generators,
                      metrics: Optional[List[ScoreMetric]] = None,
                      n_repeat: int = 100,
                      seq_len: int = 10_000,
                      alpha: float = 0.05) -> Dict:
    """
    Fonction améliorée pour comparer deux générateurs.

    Retourne un rapport détaillé de comparaison.
    """
    metrics = metrics or DEFAULT_METRICS

    # Évaluation des deux générateurs
    eval1 = evaluate_generator(gen1, n_repeat=n_repeat, seq_len=seq_len, alpha=alpha)
    eval2 = evaluate_generator(gen2, n_repeat=n_repeat, seq_len=seq_len, alpha=alpha)

    # Calcul des scores
    comparison = {
        "generator1": str(gen1),
        "generator2": str(gen2),
        "parameters": {
            "n_repeat": n_repeat,
            "seq_len": seq_len,
            "alpha": alpha
        },
        "metrics": [],
        "winner": None,
        "score1": 0,
        "score2": 0
    }

    for metric in metrics:
        val1 = metric.compute(eval1.stats)
        val2 = metric.compute(eval2.stats)

        is_better = (val1 < val2) if metric.lower_is_better else (val1 > val2)
        winner = "generator1" if is_better else "generator2"

        comparison["metrics"].append({
            "name": metric.name,
            "description": metric.description,
            "generator1_value": val1,
            "generator2_value": val2,
            "winner": winner,
            "difference": abs(val1 - val2),
            "weight": metric.weight
        })

        if winner == "generator1":
            comparison["score1"] += metric.weight
        else:
            comparison["score2"] += metric.weight

    # Détermination du gagnant global
    if comparison["score1"] > comparison["score2"]:
        comparison["winner"] = "generator1"
    elif comparison["score2"] > comparison["score1"]:
        comparison["winner"] = "generator2"
    else:
        comparison["winner"] = "draw"

    return comparison