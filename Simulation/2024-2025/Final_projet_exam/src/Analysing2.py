"""
Module des fonctions pour l'analyse automatique des séquences , l'evaluation automatique des générateurs

"""

import time
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional, Union
from numpy.typing import NDArray
from Tests import Tests , Chi2Test , KSTest
from Generators import Generators
from typing import  List, Callable
import os
from pathlib import Path

# Définition des classes personnalisées
@dataclass
class ScoreMetric:
    """
    Classe implémentant une métrique des tests.

    Attributs
    ---
    name : str
           nom de la métrique
    func : Callable
            fonction d'évaluation du score par rapport à la métrique
    weight : float
            point ajouté au gagnant.
    lower_is_better : bool
            indique si une valeur de score plus basse est meilleure
    """
    name :str
    func :Callable[[dict],float]
    lower_is_better :bool = True
    weight :float = 1.0
    def compute(self , result:dict)->float:
        """
        Calcule le score en appliquant la fonction de métrique au résultat

        """
        return self.weight*self.func(result)

@dataclass
class AnalysisResult:
    """
    Classe permettant l'encapsulation des résultats d'analyse d'une séquence afin de les manipuler pour :
        * créer , afficher et sauvegarder  des graphiques
        * generation et sauvegarde d'un rapport d'interpretation

    Attributs
    -------
        hist_df : DataFrame,
            historique des calculs de l'analyse
        summary : dict
            résumé de des résultats obtenus
        fig : Figure
            graphiques des evolution en faonction des fenêtres de :
            * p valeur
            *

    """
    _increment_ = 0

    @property
    def name(self) -> str:
        """Retourne le nom de l'analyse"""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Configure le nom de l'analyse"""
        self._name = value

    def __str__(self) -> str:
        """Représentation string de l'analyse"""
        return f"Analysis {self.name} with {len(self.hist)} points"

    def __init__(self ,
                 hist_df : pd.DataFrame ,
                 summary : dict,
                 fig : Optional[Figure]=None,
                 name : Optional[str] = None,
                 **kwargs):
        """
        instanciation d'un résultat d'analyse d'une séquence

        Parameters
        ----------
        hist : pd.DataFrame
            historique détaillé des de l'analyse
        name : str
            nom de l'expérience d'analyse
        summary : Dict[str , Dict]
            resultat final
        fig : Figure
            graphique d'évolution
        """

        self.hist = hist_df
        if name is None:
            self.name = "Result n° "+str(AnalysisResult._increment_)
            AnalysisResult._increment_=+1
        else:
            self.name = name
        self.summary = summary
        self.fig = fig
        for key , val in kwargs.items():
            setattr(self , key , val)

    def generate_report(self) -> str:
        """Génère un rapport d'interprétation basé sur les résultats."""
        report = f"# Rapport d'interprétation de {self.name}\n\n"
        for test_name, stats in self.summary.items():
            report += f"## Résultat pour le test **{test_name}**\n\n"
            report += f"```json\n{json.dumps({k: round(v, 4) for k, v in stats.items() if isinstance(v, float)}, indent=2)}\n```\n"
            report += "**→ Interprétation :**\n\n"
            # Interprétation des résultats
            acc = stats['accept_ratio']
            mean_p = stats['mean_p_value']
            viol = stats['strong_violations_ratio']
            report += self._interpret_results(acc, mean_p, viol)
            report += "\n---\n\n"
        return report

    def _interpret_results(self,
                           acc: float,
                           mean_p: float,
                           viol: float) -> str:
        """Interprète les résultats pour le rapport."""
        interpretation = ""
        if acc > 0.9:
            interpretation += f"- {int(acc * 100)}% des fenêtres sont conformes : très bonne stabilité locale.\n"
        elif acc > 0.7:
            interpretation += f"- {int(acc * 100)}% des fenêtres sont conformes : acceptable mais à surveiller.\n"
        else:
            interpretation += f"- Seulement {int(acc * 100)}% de conformité : forte instabilité locale.\n"
        if mean_p > 0.5:
            interpretation += f"- p-value moyenne élevée ({mean_p:.2f}) : la séquence semble globalement conforme.\n"
        elif mean_p > 0.1:
            interpretation += f"- p-value moyenne modérée ({mean_p:.2f}) : tendance à être limite.\n"
        else:
            interpretation += f"- p-value moyenne faible ({mean_p:.2f}) : forte suspicion de non-conformité.\n"
        if viol < 0.1:
            interpretation += f"- Seulement {int(viol * 100)}% de violations franches (stat_obs > stat_crit).\n"
        elif viol < 0.3:
            interpretation += f"- {int(viol * 100)}% de violations franches : zones localement critiques.\n"
        else:
            interpretation += f"- {int(viol * 100)}% de violations franches : comportement très irrégulier localement.\n"
        return interpretation

    def save_report(self, output_dir: str = ".") -> None:
        """Sauvegarde le rapport d'interprétation généré."""
        report = self.generate_report()
        path = Path(output_dir) / f"Rapport_{self.name}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

    def plot_results(self):
        """ Affichage des graphiques des résultats"""
        if self.fig:
            self.fig.show()

    def save_hist(self)->None:
        """
        Sauvegarde de hist
        """
        pass

    def save_fig(self)->None:
        """
        sauvegarde des graphiques
        """
        pass

    def save_all(self)->None:
        """
        Tout sauvegarder
        """
        self.save_report()
        self.save_hist()
        self.save_fig()

@dataclass
class EvaluationResult(AnalysisResult):
    """
    classe heritante de AnalysisResult pour l'encapsulation et la manipulation des resultats de evaluate_generator pour :
        - l'affichage et sauvegarde des graphiques
        -

    """
    def __init__(self, hist_df: pd.DataFrame, summary: dict, fig: Figure, name: str,
                 metrics: List[ScoreMetric], scores: Dict[str, float]):
        super().__init__(hist_df=hist_df, summary=summary, fig=fig, name=name)
        self.metrics = metrics
        self.scores = scores

    def compare(self, other: 'EvaluationResult') -> Dict:
        """Compare deux résultats d'évaluation"""
        comparison = {"winner": None, "scores": {}, "details": {}}

        # Trouver les métriques communes
        common_metrics = [m for m in self.metrics if m in other.metrics]

        if not common_metrics:
            raise ValueError("Aucune métrique commune pour comparer")

        wins_self = 0
        wins_other = 0

        for metric in common_metrics:
            score_self = self.scores[metric.name]
            score_other = other.scores[metric.name]

            is_self_better = (score_self < score_other) if metric.lower_is_better else (score_self > score_other)

            if is_self_better:
                wins_self += 1
                winner = "self"
            else:
                wins_other += 1
                winner = "other"

            comparison["details"][metric.name] = {
                "self_score": score_self,
                "other_score": score_other,
                "winner": winner,
                "difference": abs(score_self - score_other)
            }

        comparison["winner"] = "self" if wins_self > wins_other else "other"
        comparison["scores"] = {
            "self_wins": wins_self,
            "other_wins": wins_other
        }

        return comparison

def analyse_sequence(sequence:NDArray ,
                     tests : Optional[List[Tests]] = None,
                     alpha : float = 0.05 ,
                     granularities:Union[float,int] = None
                     )->AnalysisResult:
    """
    Analyse une séquence en profondeur en appliquant une serie de tests d'hypothèse

    Parameters
    ---
        sequence : séquence à analyser
        test : liste des tests à appliquer
        alpha : seuil de signification
        granularities : parmètres  fixant les tailles des fenêtres ; si ``NONE`` , on fait une analyse globale
        visualize : pour visualiser en console le resumé statistiques summary sous forme de tableau

    Returns
    ---
        AnalysisResult :
            Objet contenant ;
        - hist : DataFrame
            DataFrame de l'historique des resultats selon chaque test
        - summary : dict
            un résumé statistique ,  dictionnaire contenant pour chaque test ;
            * mean_p_value : la moyenne des p valeurs sur l'ensemble des tests fait sur les fenêtres
            * accept_ratio : le taux d'acceptation de l'hypothèse null sur l'ensemble des fenêtres
            * std_p_value : le std des p valeurs
            * strong_violation_ratio : le taux de reject selon ``stat_obs``>``stat_crit``
            * details : des détails supplémentaires
        - fig : Figure
            l'ensemble des graphique pour chaque test montrant :
            * p valeur vs seuil de signification au cours l'evolution dans la séquence
            * les statistiques observées (``stat_obs``) vs les statistiques critiques (``stat_crit``) au cours de l'évolution dans la séquence
    """
    if tests is None:
        tests = [Chi2Test() , KSTest()]

    n = len(sequence)

    # fixer la taille de la fenêtre wind_size
    if granularities is None :
        wind_size = n-1
        overlap = 0
    elif isinstance(granularities,int):
        wind_size = granularities
    elif isinstance(granularities , float):
        wind_size = np.floor(granularities*n)
    else:
        wind_size = granularities

    if wind_size>=n:
        raise ValueError(f"la taille de la fenêtre trop grande , diminuez la de : { int(granularities - n//2) }")
    # fixer l'overlap

    overlap = wind_size//2


    hist = {str(t):[] for t in tests}
    indices = []

    # Découper la séquence en fenêtre de taille wind_size
    for i in range(0,n-wind_size , overlap):
        window = sequence[i:i + wind_size]
        indices.append(i)

        for t in tests:
            result = t.test(data=window, alpha=alpha)
            hist[str(t)].append(result)

    summary = {}
    for test_name, res_list in hist.items():
        p_values = [r["p_value"] for r in res_list]
        acceptances = [r["accept"] for r in res_list]
        stat_obs_list = [r["stat_obs"] for r in res_list]
        stat_crit_list = [r["stat_crit"] for r in res_list]
        strong_violations = [s > c for s, c in zip(stat_obs_list, stat_crit_list)]

        summary[test_name] = {
            "mean_p_value": np.mean(p_values),
            "std_p_value": np.std(p_values),
            "accept_ratio": np.mean(acceptances),
            "mean_stat_obs": np.mean(stat_obs_list),
            "std_stat_obs": np.std(stat_obs_list),
            "mean_stat_crit": np.mean(stat_crit_list),
            "strong_violations_ratio": np.mean(strong_violations),
            "details": res_list
        }

    # Visualisation
    fig, axs = plt.subplots(len(tests), 2, figsize=(16, 4 * len(tests)), sharex=True)
    if len(tests) == 1:
        axs = np.array([axs])  # ensure 2D array

    for i, test in enumerate(tests):
        test_name = str(test)
        pvals = [r["p_value"] for r in hist[test_name]]
        stats_obs = [r["stat_obs"] for r in hist[test_name]]
        stats_crit = [r["stat_crit"] for r in hist[test_name]]

        # P-values
        axs[i, 0].plot(indices, pvals, label='p-value')
        axs[i, 0].axhline(alpha, color='red', linestyle='--', label='seuil α')
        axs[i, 0].set_title(f"{test_name} - p-values")
        axs[i, 0].set_ylabel("p-value")
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # stat_obs vs stat_crit
        axs[i, 1].plot(indices, stats_obs, label="stat_obs")
        axs[i, 1].plot(indices, stats_crit, label="stat_crit", linestyle='--')
        axs[i, 1].set_title(f"{test_name} - stat_obs vs stat_crit")
        axs[i, 1].set_ylabel("Valeur")
        axs[i, 1].legend()
        axs[i, 1].grid(True)

    plt.xlabel("Début de fenêtre")
    plt.tight_layout()

    return AnalysisResult(hist , summary , fig , name = f"{str(sequence)} analyse ")


def analyse_sequence2(sequence:NDArray ,
                     tests : Optional[List[Tests]] = None,
                     alpha : float = 0.05 ,
                     granularities:Optional[List[Union[float,int]]] = None ,
                     visualize : bool = False
                     )->AnalysisResult:
    """
    Analyse une séquence en profondeur en appliquant une serie de tests d'hypothèse

    Parameters
    ---
        sequence : séquence à analyser
        test : liste des tests à appliquer
        alpha : seuil de signification
        granularities : parmètres  fixant les tailles des fenêtres ; si ``NONE`` , on fait une analyse globale
        visualize : pour visualiser en console le resumé statistiques summary sous forme de tableau

    Returns
    ---
        AnalysisResult :
            Objet contenant ;
        - hist : DataFrame
            DataFrame de l'historique des resultats selon chaque test
        - summary : dict
            un résumé statistique ,  dictionnaire contenant pour chaque test ;
            * mean_p_value : la moyenne des p valeurs sur l'ensemble des tests fait sur les fenêtres
            * accept_ratio : le taux d'acceptation de l'hypothèse null sur l'ensemble des fenêtres
            * std_p_value : le std des p valeurs
            * strong_violation_ratio : le taux de reject selon ``stat_obs``>``stat_crit``
            * details : des détails supplémentaires
        - fig : Figure
            l'ensemble des graphique pour chaque test montrant :
            * p valeur vs seuil de signification au cours l'evolution dans la séquence
            * les statistiques observées (``stat_obs``) vs les statistiques critiques (``stat_crit``) au cours de l'évolution dans la séquence
    """
    if tests is None:
        tests = [Chi2Test() , KSTest()]

    n_test = len(tests)

    n = len(sequence)

    # prétraitement de granularities
    if granularities is None : # pas de granularité , analyse globale de la séquence
        granularities = np.array([n-1])
    elif isinstance(granularities,int): # un entier
        granularities = np.array([granularities])
    elif isinstance(granularities , float): # un pourcentage de la taille de la séquence
        granularities = [ np.floor(granularities*n)]
    elif all(isinstance(x,float) for x in granularities) : # liste de pourcentage de la taille de la séquence
        granularities = np.floor(np.array(granularities)*n)
    elif all(isinstance(x,int) for x in granularities) :
        granularities = np.array(granularities)
    else :
        raise ValueError("les granularities sont soit des int soit des float , pas les deux")

    n_gran = len(granularities)
    more_gran = np.where(granularities>=n)[0]
    if len(more_gran)>=n_gran:
        raise ValueError("les tailles de fenêtre sont trop grandes")
    # fixer l'overlap

    overlap = granularities//2
    rows = n_gran*n_test

    colum = ["Test" , "window start" , "window end" , "p-value" , "Stat_obs" , "Stat_crit" , "accept"]
    hist = pd.DataFrame(columns=colum)

    for wd , over in zip(granularities , overlap) :
        indices_wd = []
        # Découper la séquence en fenêtre de taille wind_size
        for i in range(0,n-wd , over):
            window = sequence[i:i + wd]
            indices_wd.append(i)

            for t in tests:
                res = t.test(data=window, alpha=alpha)
                hist = hist.append({
                    "Test":str(t),
                    "Window Start":i,
                    "Window end":i+wd,
                    **res
                },ignore_index=True)

        summary = {
        "mean_p_values":hist.groupby(["Test",])
        }
        for test_name, res_list in hist.items():
            p_values = [r["p_value"] for r in res_list]
            acceptances = [r["accept"] for r in res_list]
            stat_obs_list = [r["stat_obs"] for r in res_list]
            stat_crit_list = [r["stat_crit"] for r in res_list]
            strong_violations = [s > c for s, c in zip(stat_obs_list, stat_crit_list)]

            summary[test_name] = {
                "mean_p_value": np.mean(p_values),
                "std_p_value": np.std(p_values),
                "accept_ratio": np.mean(acceptances),
                "mean_stat_obs": np.mean(stat_obs_list),
                "std_stat_obs": np.std(stat_obs_list),
                "mean_stat_crit": np.mean(stat_crit_list),
                "strong_violations_ratio": np.mean(strong_violations),
                "details": res_list
            }

    # Visualisation
    fig, axs = plt.subplots(len(tests), 2, figsize=(16, 4 * len(tests)), sharex=True)
    if len(tests) == 1:
        axs = np.array([axs])  # ensure 2D array

    for i, test in enumerate(tests):
        test_name = str(test)
        pvals = [r["p_value"] for r in hist[test_name]]
        stats_obs = [r["stat_obs"] for r in hist[test_name]]
        stats_crit = [r["stat_crit"] for r in hist[test_name]]

        # P-values
        axs[i, 0].plot(indices, pvals, label='p-value')
        axs[i, 0].axhline(alpha, color='red', linestyle='--', label='seuil α')
        axs[i, 0].set_title(f"{test_name} - p-values")
        axs[i, 0].set_ylabel("p-value")
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # stat_obs vs stat_crit
        axs[i, 1].plot(indices, stats_obs, label="stat_obs")
        axs[i, 1].plot(indices, stats_crit, label="stat_crit", linestyle='--')
        axs[i, 1].set_title(f"{test_name} - stat_obs vs stat_crit")
        axs[i, 1].set_ylabel("Valeur")
        axs[i, 1].legend()
        axs[i, 1].grid(True)

    plt.xlabel("Début de fenêtre")
    plt.tight_layout()

    if visualize:
        fig.show()

    return ResultAnalysis(hist , summary , fig , name = f"{str(sequence)} analyse ")


def evaluate_generator(generator : Generators ,
                       tests:List[Tests] = None ,
                       granularities : List[Union[int , float]]=None,
                       alpha: float = 0.05 ,
                       n_repeat: int = 200,
                       sequence_lenght: int = 10_000,
                       metrics : List[ScoreMetric]=None,
                       verbose:bool=False
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
    sequence_lenght : int, optional
        taille des séquences générées à chaque itération
    metrics : List[ScoreMetric], optional
        métriques pour l'évaluation
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    EvaluationResult :
        Objet contenant ;
    - hist : DataFrame
        DataFrame de l'historique des resultats selon chaque test
    - summary : dict
        un résumé statistique ,  dictionnaire contenant pour chaque test ;
        * mean_p_value : la moyenne des p valeurs sur l'ensemble des tests fait sur les fenêtres
        * accept_ratio : le taux d'acceptation de l'hypothèse null sur l'ensemble des fenêtres
        * std_p_value : le std des p valeurs
        * strong_violation_ratio : le taux de reject selon ``stat_obs``>``stat_crit``
        * details : des détails supplémentaires
    - fig : Figure
        l'ensemble des graphique montrant pour chaque test :
        * p valeur vs seuil de signification au cours l'evolution dans la séquence
        * les statistiques observées (``stat_obs``) vs les statistiques critiques (``stat_crit``) au cours de l'évolution dans la séquence
    - metrics : List[ScoreMetric]
        liste des métriques à calculées
    - scores : dict
        scores du génrérateur pour chacune des métriques
    """

    if tests is None:
        tests = [Chi2Test() , KSTest()]
    n_test = len(tests)


    if granularities is None :
        n_gran = 1
    n_gran = len(granularities)
    rows = n_repeat*n_test*n_gran
    all_results = []
    total_violations = 0
    total_accept_ratio = 0.0
    per_test_stats = {}

    for i in range(n_repeat):
        sequence = generator.generate(sequence_lenght)
        result = analyse_sequence(sequence, tests, granularities, alpha)
        all_results.append(result)

        total_accept_ratio += result["summary"]["accept_ratio"]
        total_violations += result["summary"]["strong_violations"]

        for test_name, stat in result["summary"]["by_test"].items():
            if test_name not in per_test_stats:
                per_test_stats[test_name] = {"n_passed": 0, "n_total": 0}
            per_test_stats[test_name]["n_passed"] += stat["n_passed"]
            per_test_stats[test_name]["n_total"] += stat["n_total"]

    # Calculs finaux
    mean_accept_ratio = total_accept_ratio / n_repeat
    test_summary = {
        name: {
            "acceptance_rate": stat["n_passed"] / stat["n_total"] if stat["n_total"] else 0.0,
            "total_tests": stat["n_total"]
        }
        for name, stat in per_test_stats.items()
    }

    return {
        "n_repeats": n_repeat,
        "sequence_length": sequence_lenght,
        "granularity": granularities,
        "mean_accept_ratio": mean_accept_ratio,
        "total_strong_violations": total_violations,
        "per_test_summary": test_summary,
        "all_results": all_results  # Peut être ignoré pour résumé
    }


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

# Nouvelles métriques
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
        func=lambda x : x ,
        lower_is_better = False
    )
]
