"""
Module des fonctions pour l'analyse automatique des séquences , l'evaluation automatique des générateurs

"""


import json
from dataclasses import dataclass , field
from pathlib import Path
from typing import List, Dict, Optional, Union , Callable
import numpy as np
from pandas import DataFrame , concat
import plotly.express as px
from numpy.typing import NDArray
from Tests import Tests , Chi2Test , KSTest
from Generators import Generators


# Définition des classes personnalisées
@dataclass
class ScoreMetric:
    """
    Classe implémentant une métrique de test
    """
    name :str # nom de la métrique
    func :Callable[[dict],float] # évaluation du score par rapport à la métrique
    lower_is_better :bool = True  # valeur de score plus basse est meilleure
    weight :float = 1.0 # point ajouté au gagnant
    def compute(self , result:dict)->float:
        """
        Calcule le score en appliquant la fonction de métrique au résultat

        """
        return self.weight*self.func(result)

DEFAULT_METRICS = [
    ScoreMetric(
        name="convergence",
        func=lambda x: x.get("convergence_score", 0),
        weight=2.0
    ),
    ScoreMetric(
        name="stability",
        func=lambda x: x.get("stability_score", 0),
        weight=1.5
    ),
    ScoreMetric(
        name="mean_p_value",
        func=lambda r: r.stats["mean_p_value"],
        lower_is_better=False,
        weight=1.5
    ),
    ScoreMetric(
        name="accept_ratio",
        func=lambda r: r.stats["accept_ratio"],
        lower_is_better=False,
        weight=2.0
    ),
    ScoreMetric(
        name="violation_ratio",
        func=lambda r: r.stats["violation_ratio"],
        lower_is_better=True,
        weight=1.0
    ),
    ScoreMetric(
        name="std_p_value",
        func=lambda r: r.stats["std_p_value"],
        lower_is_better=True,
        weight=0.5
    )
]


@dataclass
class AnalysisResult:
    """
    Classe permettant l'encapsulation des résultats d'analyse d'une séquence afin de les manipuler pour :
        * créer , afficher et sauvegarder  des graphiques
        * generation et sauvegarde d'un rapport d'interpretation

    Attributs
    -------
    hist_df (DataFrame):
            historique des calculs de l'analyse de la séquence
    stats (dict):
            résumé aggrégé des statistiques
    window_sizes (List[int]):
            liste des taille de sous séquence utilisées


    """

    hist_df : DataFrame
    stats : Dict[str,float]
    name : str
    window_sizes : List[int]
    _increment : int = 0

    def __post_init__(self):
        if not self.name:
            self.name = f"Result_{self.__class__._increment}"
        self.__class__._increment += 1

    def __str__(self) -> str:
        """Représentation string de l'analyse"""
        return self.name

    @property
    def summary_by_test(self)->DataFrame:
        """ aggregation des résultats par test"""
        return self.hist_df.copy().groupby("Test").agg({
            "p_value":["mean" , "median","std"],
            "accept":"mean",
            "stat_obs":"mean"
        }).rename(columns={"mean":"mean_","std":"std_","median":"median_"})

    @property
    def summary_by_granularity(self)->DataFrame:
        """ aggregation des résultats par granularité"""
        return self.hist_df.copy().groupby("window_size").agg({
            "p_value":["mean" , "median","std"],
            "accept":"mean"
        })

    def generate_report(self) -> str:
        """
        Génère un rapport au format Markdown.
        """
        report = f"# Rapport d'analyse : {self.name}\n\n"

        # Statistiques globales
        report += "## Statistiques globales\n\n"
        stats = self.stats

        # Convertir les valeurs numpy.float64 en float Python standard
        formatted_stats = {
            k: float(v) if isinstance(v, np.float64) else v
            for k, v in stats.items()
        }

        # Arrondir les valeurs flottantes
        rounded_stats = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in formatted_stats.items()
        }

        report += f"```json\n \n{json.dumps(rounded_stats, indent=2)}\n```\n\n"

        report+=  self._interpret_stats(rounded_stats)

        return report

    def _interpret_stats(self, stats: dict) -> str:
        """Interprétation des statistiques"""
        report = "## Interprétation des stats :\n\n"

        # Analyse de la p-value moyenne
        mean_p = stats.get('mean_p_value', 0.5)
        if mean_p > 0.7:
            report += f"- Excellente conformité (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.5:
            report += f"- Bonne conformité (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.3:
            report += f"- Conformité modérée (p-value moyenne: {mean_p:.3f})\n"
        elif mean_p > 0.1:
            report += f"- Conformité faible (p-value moyenne: {mean_p:.3f})\n"
        else:
            report += f"- Non-conformité (p-value moyenne: {mean_p:.3f})\n"

        # Analyse du taux d'acceptation
        acc_ratio = stats.get('accept_ratio', 0)
        alpha = stats.get('alpha', 0.05)
        expected_ratio = 1 - alpha

        if acc_ratio > expected_ratio * 1.1:
            report += f"- Taux d'acceptation anormalement élevé ({acc_ratio:.1%} vs {expected_ratio:.1%} attendu)\n"
        elif acc_ratio > expected_ratio * 0.9:
            report += f"- Taux d'acceptation normal ({acc_ratio:.1%})\n"
        else:
            report += f"- Taux d'acceptation trop bas ({acc_ratio:.1%} vs {expected_ratio:.1%} attendu)\n"

        return report


    def save_all(self, output_dir: Union[str, Path] = "results") -> None:
        """Sauvegarde tous les résultats dans un dossier."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self.save_report(output_path)
        self.save_hist(output_path)
        #self.save_fig(output_path)

    def save_report(self, output_path: Path) -> None:
        """Sauvegarde le rapport."""
        report_path = output_path / f"report_{self.name}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.generate_report())

    def save_hist(self, output_path: str) -> None:
        """Sauvegarde l'historique des résultats."""
        output_path = Path(output_path)
        hist_path = output_path / f"history_{self.name}.csv"
        self.hist_df.to_csv(hist_path, index=False)

    def plot_p_values_evolution(self):
        """
        Trace l'évolution des p-values pour chaque test selon la granularité.
        """
        fig = px.line(self.hist_df, x="window_size", y="p_value", color="Test",
                      title=f"Évolution des p-values pour {self.name}")
        fig.show()

    def plot_p_values_distribution(self):
        """
        Affiche la distribution des p-values (violin plot) pour chaque test.
        """
        fig = px.violin(self.hist_df, y="p_value", color="Test", box=True, points="all",
                        title=f"Distribution des p-values pour {self.name}")
        fig.show()

    def plot_p_values_box_by_granularity(self):
        """
        Affiche les boîtes à moustaches des p-values par taille de fenêtre.
        """
        fig = px.box(self.hist_df,
                     x="window_size",
                     y="p_value",
                     color="Test",
                     title=f"P-values par granularité pour {self.name}")
        fig.show()

    def plot_p_values_box_by_test(self):
        """
        affiche les boîtes à moustaches des p-values par test.
        """
        fig = px.box(self.hist_df,
                     x = "Test",
                     y ="p_value",
                     color = "window_size",
                     title = f"p-values par test pour {self.name}")
        fig.update_xaxes(title_text="Tests")
        fig.update_yaxes(title_text="p_value")
        fig.show()

    def plot_heatmap_acceptance(self)->None:
        """
        Affiche une heatmap des taux d'acceptation des tests selon la granularité.
        """
        pivot = self.hist_df.pivot_table(index="window_size",
                                         columns="Test",
                                         values="accept",
                                         aggfunc="median")
        fig = px.imshow(pivot , text_auto=True,aspect="auto",
                        title=f"Taux d'acceptation par granularité -{self.name}")
        fig.update_xaxes(title_text="Tests")
        fig.update_yaxes(title_text="Taille de la fenêtre")
        fig.show()

    def plot_heatmap_p_values(self)->None:
        """
        Affiche une heatmap des p-values selon la granularité.
        """
        pivot = self.hist_df.pivot_table(index="window_size",
                                         columns="Test",
                                         values="p_value",
                                         aggfunc="median")
        fig = px.imshow(pivot , text_auto=True,aspect="auto",
                        title=f"p-values par granularité -{self.name}")
        fig.update_xaxes(title_text="Tests")
        fig.update_yaxes(title_text="Taille de la fenêtre")
        fig.show()

class EvaluationResult(AnalysisResult):
    """
    Classe heritante de 'AnalysisResult' permettant l'encapsulation des résultats d'évaluation des générateurs
    """
    def __init__(self,
                 hist_df: DataFrame,
                 stats: dict,
                 generator_name: str,
                 window_sizes: List[int] = None,
                 **kwargs):
        super().__init__(hist_df, stats ,generator_name, window_sizes=window_sizes)
        self.metrics = kwargs.get("metrics", DEFAULT_METRICS)


    def generate_report(self) -> str:
        report = super().generate_report()
        report += "## Analyse par granularité\n"
        for w in self.window_sizes:
            subset = self.hist_df[self.hist_df["window_size"] == w]
            report += f"- Fenêtre de taille {w} : {subset['accept'].mean():.2%} de conformité\n"
        report += "\n## Résumé par test\n"
        report += self.summary_by_test+"\n\n"
        return report


    def plot_metric_comparison(self, other: "EvaluationResult", metric: str):
        """Comparaison de métrique entre 2 générateurs"""
        df = DataFrame({
            "Générateur": [self.name, other.name],
            metric: [self.stats[metric], other.stats[metric]]
        })
        px.bar(df, x="Générateur", y=metric, title=f"Comparaison de {metric}").show()




def to_wind_size(granularities: Union[List[Union[int,float]], int, float],
                 len_seq: int
                 )->List[int]:
    """
    Transforme une granularité ou liste de granularités en taille(s) entière(s) de sous séquence
    """
    if len_seq<=0:
        raise ValueError("la taillle de la séquence doit être > 0")
    # Si c'est un scalaire, le convertir en liste
    if isinstance(granularities, (int, float)):
        granularities = [granularities]
    # Vérifier si les granularités sont des entiers ou des fractions
    elif np.all([isinstance(g,int) and 0<g<=len_seq for g in granularities]) :
        return granularities
    elif np.all([isinstance(g,float) and 0.0<g<=1.0 for g in granularities]):
        return [int(g*len_seq) for g in granularities]
    else :
        raise ValueError(f"les granularités doivent être > 0 ,  être soit des entiers <={len_seq} , soit des fractions  <=1 de la taille de ")


def analyse_sequence(sequence:NDArray ,
                     tests : Optional[List[Tests]] = None,
                     alpha : float = 0.05 ,
                     granularities:Union[List[Union[int,float]], int, float] = None,
                     name : str = None
                     )->AnalysisResult:
    """
    Analyse une séquence en profondeur en appliquant une serie de tests statistiques

    Parameters
    ---
        sequence : séquence à analyser
        test : liste des tests à appliquer
        alpha : seuil de signification
        granularities : parmètres  fixant les tailles des fenêtres ; si ``NONE`` , on fait une analyse globale
        name : titre de l'analyse (pour les tableaux des résultats et les graphiques)
    Returns
    ---
        AnalysisResult :
            Objet contenant ;
        - hist : DataFrame
            DataFrame de l'historique des resultats selon chaque test
        - stats : dict
            un résumé statistique ,  dictionnaire contenant pour chaque test ;

            - mean_p_value : la moyenne des p valeurs sur l'ensemble des tests fait sur les fenêtres
            - accept_ratio : le taux d'acceptation de l'hypothèse null sur l'ensemble des fenêtres
            - std_p_value : le std des p valeurs
            - violation_ratio : le taux de reject selon ``stat_obs``>``stat_crit``
            - rejection_ratio : le taux de rejet de l'hypothèse null sur l'ensemble des fenêtres
    """
    if tests is None:
        tests = [Chi2Test() , KSTest()]

    n = len(sequence)
    if granularities is None:
        granularities = [1.0]
    # fixer les tailles des fenêtre wind_sizes
    window_sizes = to_wind_size(granularities,n)
    results = []
    for test in tests:
        for w in window_sizes:
            # Découper la séquence en sous séquence de taille wind_size
            for i in range(0,n-w +1 , w):
                sub_seq = sequence[i:i+w]
                test_result = test.test(sub_seq, alpha)
                res = {
                "Test" : str(test),
                "window_start" : i,
                "window_size" : w,
                "p_value" : test_result.p_value,
                "stat_obs" : test_result.stat_obs,
                "stat_crit" : test_result.stat_crit,
                "accept" : test_result.accept
                }
                results.append(res)
    columns = ["Test", "window_size" , "window_start" , "p_value", "accept","stat_obs","stat_crit"]
    hist_df = DataFrame(results,columns=columns)


    stats = {
        "mean_p_value": hist_df["p_value"].mean(),
        "median_p_value":hist_df["p_value"].median(),
        "accept_ratio": hist_df["accept"].mean(),
        "rejection_ratio": 1 - hist_df["accept"].mean(),
        "std_p_value":hist_df["p_value"].std(),
        "alpha": alpha
    }

    return AnalysisResult(hist_df=hist_df ,
                          stats=stats,
                          window_sizes=window_sizes,
                          name=name)


def evaluate_generator(generator : Generators ,
                       tests:Optional[List[Tests]] = None,
                       alpha: float = 0.05 ,
                       n_repeat: int = 200,
                       seq_len: int = 10_000
                        )->EvaluationResult:
    """
    Evalue le générateur en générant n_repeat fois des séquences de taille sequence_lenght et en les analysant

    Parameters
    ----------
    generator : Generators
        générateur à evaluer
    tests : List[Tests], optional
        test à appliquer
    granularities : List[Union[int , float]], optional
        tailles des fenêtres
    alpha : float, optional
        seuil de signification
    n_repeat : int, optional
        nombre de repetition
    seq_lenght : int, optional
        taille des séquences générées à chaque itération


    Returns
    -------
    EvaluationResult :
        Objet contenant ;
    - hist : DataFrame
        DataFrame de l'historique des resultats selon chaque test
    - stats : dict
        un résumé statistique ,  dictionnaire contenant pour chaque test ;
        * mean_p_value : la moyenne des p valeurs sur l'ensemble des tests fait sur les fenêtres
        * accept_ratio : le taux d'acceptation de l'hypothèse null sur l'ensemble des itérations
        * std_p_value : le std des p valeurs
        * violation_ratio : le taux de violation selon ``stat_obs``>``stat_crit``
        * rejection_ratio le taux de rejet de l'hypothèse null sur l'ensemble des itérations


    """

    if tests is None:
        tests = [Chi2Test() , KSTest()]

    granularities = [0.1,0.3,0.5,1.0]
    window_size = to_wind_size(granularities , seq_len)
    all_results = []
    for i in range(n_repeat):
        seq = generator.generate(seq_len)
        res = analyse_sequence(sequence=seq ,
                               alpha=alpha ,
                               tests=tests , granularities=granularities)
        res.hist_df["Iteration"]=i
        all_results.append(res.hist_df)

    df = concat(all_results,ignore_index=True)

    stats = {
        "mean_p_value":df["p_value"].mean(),
        "std_p_value":df["p_value"].std(),
        "accept_ratio":df["accept"].mean(),
        "rejection_ratio": 1 - df["accept"].mean(),
        }

    return EvaluationResult(hist_df=df ,
                            generator_name=str(generator),
                            stats=stats,
                            window_sizes=window_size
            )


def enhanced_score( gen1: Generators,
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
        total_points+=1
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


def analyse_convergence(generator: Generators,
                       tests: List[Tests],
                       alpha: float = 0.05,
                       n_range: List[int] = None,
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

    if n_range is None :
        n_range = [1000, 5000, 10000, 50000, 100000]
    for n in n_range:
        results = evaluate_generator(generator, tests=tests, alpha=alpha, n_repeat=k, seq_len=n).stats

        # Calculer les métriques de convergence
        rejection_rate = results['rejection_ratio'].mean()
        p_value_stability = 1 - results['std_p_value'].mean()

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


def compare_p_values(gen1_result: EvaluationResult,
                     gen2_result: EvaluationResult):
    """
    Compare les p-values par test entre deux générateurs (boîtes à moustaches).
    """
    df1 = gen1_result.hist_df.copy()
    df1["Générateur"] = gen1_result.name
    df2 = gen2_result.hist_df.copy()
    df2["Générateur"] = gen2_result.name
    df = DataFrame([*df1.to_dict("records"), *df2.to_dict("records")])

    fig = px.box(df, x="Test", y="p_value", color="Générateur",
                 title="Comparaison des p-values par test entre les deux générateurs")
    fig.show()


def compare_summary_stats(gen1_result: EvaluationResult, gen2_result: EvaluationResult, stat: str):
    """
    Compare une statistique globale (accept_ratio, mean_p_value, etc.) entre deux générateurs.
    """
    df = DataFrame({
        "Générateur": [gen1_result.name, gen2_result.name],
        stat: [gen1_result.stats[stat], gen2_result.stats[stat]]
    })
    fig = px.bar(df, x="Générateur", y=stat, title=f"Comparaison de la statistique '{stat}'")
    fig.show()


