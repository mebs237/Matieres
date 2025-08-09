"""
Module des fonctions pour l'analyse des séquences , l'evaluation des générateurs

"""

from abc import ABC , abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union , Literal
from enum import Enum
from collections import namedtuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import norm
from .Tests import Tests , Chi2Test
from .Generators import Generators
from .Utils import to_wind_size , requires__run

ComputeRes = namedtuple("res_compute", ["hist_df", "stats", "window_sizes"])

def eval_test_power(test_class,
                    n: int = 500,
                    k: int = 100,
                    alpha: float = 0.05,
                    generator_null=None,
                    generator_alt=None,
                    verbose=True,
                    **test_kwargs):
    """
    Évalue la puissance et le comportement (taux de faux positifs et faux négatifs)
    d'un test statistique personnalisé.

    Args:
        test_class: Classe du test à évaluer (ex: Chi2Test)
        n: Taille des échantillons à générer
        k: Nombre d'itérations pour estimer les taux
        alpha: Niveau de signification
        generator_null: Callable pour générer des données sous H0 (sinon np.random.uniform)
        generator_alt: Callable pour générer des données sous H1 (sinon np.random.normal)
        verbose: Affichage des résultats
        test_kwargs: Paramètres supplémentaires pour initialiser le test

    Returns:
        dict: Résultats complets
    """
    if generator_null is None:
        generator_null = lambda n: np.random.uniform(0, 1, n)
    if generator_alt is None:
        generator_alt = lambda n: np.clip(np.random.normal(0.5, 0.1, n), 0, 1)

    test = test_class.__class__(**test_kwargs)
    faux_positifs = 0
    rejets_H1 = 0

    for _ in range(k):
        data_H0 = generator_null(n)
        result_H0 = test.run_test(data_H0, alpha=alpha)
        if not result_H0.accept:
            faux_positifs += 1

        data_H1 = generator_alt(n)
        result_H1 = test.run_test(data_H1, alpha=alpha)
        if not result_H1.accept:
            rejets_H1 += 1

    taux_fp = faux_positifs / k
    puissance = rejets_H1 / k
    taux_fn = 1 - puissance

    resume = {
        "test": str(test),
        "n": n,
        "k": k,
        "alpha": alpha,
        "taux_faux_positifs (doit ≈ alpha)": taux_fp,
        "puissance (doit ≈ 1)": puissance,
        "taux_faux_négatifs": taux_fn
    }

    if verbose:
        print("\n Évaluation de la puissance et rigueur du test :", str(test))
        for key, val in resume.items():
            print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    return resume

class Status(Enum):
    """status possibles pour l'intervalle de confiance"""
    Normal = "normal"
    High = "high"
    Low = "low"

class level(Enum):
    """niveaux d'abstraction possibles pour l'analyse d'une séquence ou un générateur"""
    Global = "Global"
    Bytest = "Test"
    Bysize = "Size"
    Byiter = "Iteration"

# niveau d'abstraction pour l'analyse des séquences
seqlevel = [level.Global, level.Bytest, level.Bysize]

def analyse_sequence(sequence: NDArray,
                     tests: Optional[List[Tests]] = None,
                     alpha: float = 0.05,
                     granularities: Optional[List[Union[int, float]]] = None) -> ComputeRes:
    """
    fait les calculs de l'analyse de la séquence
    """

    tests = tests if tests else [Chi2Test()]

    n = len(sequence)

    granularities = granularities if granularities else [1.0]
    # fixer les tailles des fenêtre wind_sizes
    window_sizes = to_wind_size(granularities,n)
    results = []

    for test in tests:
        for w in window_sizes:
            # Découper la séquence en sous séquence de taille wind_size
            for i in range(0,n-w +1 , w):
                sub_seq = sequence[i:i+w]
                test_result = test.run_test(sub_seq, alpha)
                res = {
                "Test" : str(test),
                "Start" : i,
                "Size" : w,
                "p_value" : test_result.p_value,
                "stat_obs" : test_result.stat_obs,
                "stat_crit" : test_result.stat_crit,
                "accept" : test_result.accept
                }
                results.append(res)

    columns = ["Test", "Size" , "Start" , "p_value", "accept","stat_obs","stat_crit"]
    hist_df = pd.DataFrame(results,columns=columns)

    stats = {
        "mean_p_value": hist_df["p_value"].mean(),
        "std_p_value": hist_df["p_value"].std(),
        "mean_accept": hist_df["accept"].mean(),
        "std_accept": hist_df["accept"].std(),
    }

    return ComputeRes(hist_df , stats , window_sizes)

def evaluate_generator(generator : Generators ,
                       tests:Optional[List[Tests]] = None,
                       alpha: float = 0.05 ,
                       n_repeat: int = 200,
                       seq_len: int = 10_000,
                       granularities: Optional[List[Union[int, float]]] = None,
                        )->ComputeRes:
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
        Objet contenant ;
    - hist : pd.DataFrame
        DataFrame de l'historique des resultats selon chaque test
    - stats : dict
        un résumé statistiques globales
    """
    tests = tests if tests else [Chi2Test()]
    all_results = []
    for i in range(n_repeat):
        seq = generator.generate(seq_len)
        res = analyse_sequence(sequence=seq ,
                               alpha=alpha ,
                               tests=tests , granularities=granularities)
        res.hist_df["Iteration"]=i
        all_results.append(res.hist_df)

    df = pd.concat(all_results,ignore_index=True)
    columns_order = ["Iteration", "Test", "Size", "Start", "p_value", "accept", "stat_obs", "stat_crit"]
    df = df[columns_order].set_index("Iteration")

    stats = {
        "mean_p_value":df["p_value"].mean(),
        "std_p_value":df["p_value"].std(),
        "mean_accept":df["accept"].mean(),
        "std_accept":df["accept"].std()
        }

    return ComputeRes(hist_df=df,stats=stats,
                      window_sizes=res.window_sizes)

class Analyzer(ABC):
    """
    Interface de classe  d'analyse  :
        * exécute les calculs des tests et les interpètent
        * créer , afficher et sauvegarder  des graphiques
        * generation et sauvegarde d'un rapport d'interpretation

    Attributs
    -------
    tests:
        liste des tests à appliquer
    alpha :
        seuil de signification
    name :
        titre de l'analyse (pour les tableaux et les graphiques)
    hist_df :
        tableau contenant l'historique des calculs

    """

    def __init__(self,
                 tests: Optional[List[Tests]] = None,
                 granularities: Optional[List[Union[int, float]]] = None,
                 alpha: float = 0.05):
        self.tests = tests
        self.granularities = granularities
        self.alpha = alpha
        self.runed = False
        self.hist_df = None

    @abstractmethod
    def run(self):
        """Exécute les calculs des tests pour l'évaluation."""

    @abstractmethod
    def suffix(self):
        """suffixe pour les noms de fichiers et graphiques"""
        return f"analyse_alpha={self.alpha}"

    @requires__run
    def stats_group_by(self,
                       by:Union[level , List[level]]=level.Global
                       ) -> pd.DataFrame:
        """
        Agrège les valeurs de dataframe selon un ou plusieurs niveau :
        - "Test", "size" , "Iteration" , combinaison de deux , Global
        """

        df = self.hist_df.copy()

        if isinstance(by , level):
            by = [by]
        elif isinstance(by, list):
            if any(not isinstance(b,level) for b in by):
                raise ValueError("on doit avoir des niveaux ")
        else:
            raise ValueError("on doit avoir des niveaux ")


        if level.Global in by:
            # Agrégation globale
            stats = {
            'mean_p_value': df['p_value'].mean(),
            'mean_accept': df['accept'].mean(),
            'std_accept': df['accept'].std(),
            'std_p_value': df['p_value'].std(),
            'count': df.shape[0]
            }
            return pd.DataFrame([stats], index=['Global'])
        else:
            by = [b.value for b in by]
            # Agrégation par la/les colonnes spécifiées
            stats = df.groupby(by).agg(
                mean_p_value=("p_value", "mean"),
                mean_accept=("accept", "mean"),
                std_accept=("accept", "std"),
                std_p_value=("p_value", "std"),
                count=("accept","count")
                ).reset_index().sort_values(by).set_index(by)

        return stats

    def interpret_stats(self,
                        stats: Dict[str, Union[float,int]],
                        detail:bool=True) -> str:
        """interprète les statistiques ``stats``"""
        report = []
        alpha = self.alpha
        n_tests = stats.get("count")
        n_root = np.sqrt(n_tests)
        z = norm.ppf(1 - alpha / 2)

        # Définition des métriques et de leurs paramètres
        metrics = {
            'mean_p_value': {
                'ideal': 0.5,
                'message': "- P-value moyenne : {value:.3f} (attendu {ideal} , IC à {confidence:.0%} : [{lower:.3f}, {upper:.3f}])",
                'status_messages': {
                    Status.High: "- P-value élevée : sur-ajustement probable ou tests biaisés (trop permissifs)",
                    Status.Low: "- P-value faible : rejet de l'hypothèse (ou test trop strict)",
                    Status.Normal: "- P-value conforme aux attentes"
                }
            },
            'mean_accept': {
                'ideal': 1 - alpha,
                'message': "- Taux d'accept moyen : {value:.1%} (attendu {ideal:.0%} IC à {confidence:.0%} : [{lower:.1%}, {upper:.1%}])",
                'status_messages': {
                    Status.High: "- Taux d'accept moyen élevé : sur-ajustement probable ou test(s) biaisé(s) (trop permissifs)",
                    Status.Low: "- Taux d'accept faible : rejet de l'hypothèse (ou test trop strict)",
                    Status.Normal: "- Taux d'acceptation conforme aux attentes"
                }
            }
        }

        status_results = {}


        for metric, params in metrics.items():
            value = stats.get(metric, params['ideal'])
            std = stats.get(f'std_{metric.replace('mean_','')}', 0.0)
            lower = max(0 , value - z * std / n_root)
            upper = min(1 , value + z * std / n_root)

            if detail :
                report.append(params['message'].format(
                value=value,
                confidence=1 - alpha,
                lower=lower,
                upper=upper,
                ideal=params['ideal']
            ))

            # Détermination du statut
            if lower > params['ideal']:
                status = Status.High
            elif upper < params['ideal']:
                status = Status.Low
            else:
                status = Status.Normal

            if detail:
                report.append(params['status_messages'][status])
            status_results[metric] = status

        # Gestion des conclusions globales
        conclusion_messages = {
            (Status.Normal, Status.Normal): "On accepte l'uniformité",
            (Status.High, Status.High): "Sur-ajustement probable ou le/l'un des test(s) est(sont) biaisé(s)",
            (Status.Low, Status.Low): "On rejette l'hypothèse",
            (Status.Low, Status.Normal): "Rejet de l'hypothèse selon la p-valeur, mais acceptation selon le taux d'accept",
            (Status.Normal, Status.Low): "Rejet de l'hypothèse selon le taux d'accept, mais acceptation selon la p-valeur",
            (Status.High, Status.Normal): "Acceptation globale de l'hypothèse mais sur-ajustement ou test biaisé selon la p-valeur",
            (Status.Normal, Status.High): "Acceptation globale de l'hypothèse mais sur-ajustement ou test biaisé selon le taux d'accept",
            (Status.Low, Status.High): "rejet de l'hypothèse selon la p-valeur et sur-ajustement ou test biaisé selon le taux d'accept",
            (Status.High, Status.Low): "rejet de l'hypothèse selon le taux d'accept et sur-ajustement ou test biaisé selon la p-valeur"
        }

        p_status = status_results['mean_p_value']
        acc_status = status_results['mean_accept']
        report.append(f"- Conclusion : {conclusion_messages.get((p_status, acc_status), 'Cas non prévu')}")

        return "\n".join(report)

    @requires__run
    def report(self,lvl:Union[level,List[level]]=level.Global,
                detail:bool=False,tables:bool=False)->str:
        """verdict de l'analyse selon le niveau d'abstration ``lvl`` souhaité"""
        if isinstance(lvl,level):
            lvl = [lvl]
        sections = [f"## Rapport Analyse {self.suffix()}"]
        for lv in lvl:
            if lv == level.Global:
                sections.append(f"### synthèse {lv.value}\n")
                df = self.stats_group_by()
                if tables:
                    sections.append(df.to_markdown())
                stat = next(iter(df.to_dict(orient="index").values()))
                sections.append(self.interpret_stats(stats=stat,detail=True))
            else:
                sections.append(f"### synthèse par {lv.value}\n")
                df = self.stats_group_by(lv)
                for idx , stat in df.to_dict(orient="index").items():
                    name = idx if isinstance(idx,str) else str(idx)
                    sections.append(f"##### pour {lv.value} : {name}\n")
                    sections.append(self.interpret_stats(stats=stat,detail=detail))
        return "\n".join(sections)

    @requires__run
    def barplot_by(self,
                   by:Union[level,List[level]]= level.Bytest,
                   stat:Literal["mean_accept","mean_p_value"] = "mean_accept",
                   show=True , save_path = None):
        """ graphique en barres de ``stat`` par ``by``."""

        df = self.stats_group_by(by)
        if isinstance(by,level):
            by=[by]
        elif isinstance(by,list):
            if all(not isinstance(b,level) for b in by):
                raise ValueError(f"{by} n'est pas une valeur dans  {[l for l in level]}")
            elif len(by)>2:
                raise ValueError("les levels by doivent être au plus deux")
        else:
            raise ValueError("format des entrées invalide")

        plt.figure(figsize=(12, 8))
        if len(by)==2:
            colx , coly = by[0].value , by[1].value
            sns.barplot(data=df, x=colx , y = stat , hue=coly)
            plt.title(f"{stat} par {colx} et {coly} - {self.suffix()}")
        else:
            sns.barplot(data=df ,x=by[0].value,y=stat)
            plt.title(f"{stat} par {by[0].value} - {self.suffix()}")
        plt.ylabel(f"{stat}")
        plt.ylim(0, 1)
        plt.xticks(rotation=5)
        if save_path :
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    @requires__run
    def heatmap_by(self,
                   metric:Literal["accept","p_value"]="accept",
                   by:List[level]=None,
                   func:Literal["mean","median","std"]="mean",
                   show=True, save_path=None):
        """Affiche une heatmap de ``metric`` par test et granularité."""

        by = by if by is not None else [level.Bytest , level.Bysize]

        if len(by)!=2:
            raise ValueError("by doit être une liste de deux éléments")
        if any(not isinstance(lv,level)for lv in by):
                raise ValueError("valeurs non valides")

        colx , coly = by[0].value , by[1].value
        df = self.hist_df.copy().dropna()
        df['accept'] = df['accept'].astype(int)

        pivot = df.pivot_table(index = coly,
                               columns= colx,
                               values= metric,
                               aggfunc = func)
        plt.figure(figsize=(10,8))
        sns.heatmap(pivot, annot=True, fmt=".1%"if metric=="accept" else ".2f",vmin=0, vmax=1 , cmap='viridis')
        plt.title(f" {func} {metric} par {colx} et {coly} - {self.suffix()}")
        plt.xticks(rotation=5)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    @requires__run
    def boxplot_by(self,
                   metric:Literal["p_value","accept"]="p_value",
                   by:Union[level,List[level]]=level.Bytest,
                   show=True, save_path=None):
        """ Affiche un boxplot des p-valeur moyenne par ``by``."""
        # Supprimer les nan et remplacer les booléens
        df = self.hist_df.copy().dropna()
        df['accept'] = df['accept'].astype(int)

        if isinstance(by,level):
            by=[by]
        elif isinstance(by,list):
            if all(not isinstance(b,level) for b in by):
                raise ValueError(f"{by} n'est pas une valeur dans  {[l for l in level]}")
            elif len(by)>2:
                raise ValueError("les levels by doivent être au plus deux")
        else:
            raise ValueError("format des entrées invalide")

        plt.figure(figsize=(12, 8))
        if len(by)==2:
            colx , coly = by[0].value , by[1].value
            sns.boxplot(data=df, x=colx , y = metric , hue=coly)
            plt.title(f"{metric} par {colx} et {coly} - {self.suffix()}")
        else:
            sns.boxplot(data=df ,x=by[0].value,y=metric)
            plt.title(f"{metric} par {by[0].value} - {self.suffix()}")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.xticks(rotation=7)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def save_all(self, output_dir: Union[str, Path] = None):
        """
        Sauvegarde tous les résultats dans un dossier.
        """
        output_dir = f"Results_{self}" if output_dir is None else output_dir
        output_path = Path(output_dir or f"Results_{self.suffix()}")
        output_path.mkdir(exist_ok=True)
        self.save_report(output_path)
        self.save_stats(output_path)
        self.save_figures(output_path)

    def save_report(self, output_path: Path,
                    lvl:Union[level,List[level]]=None) -> None:
        """Sauvegarde le rapport."""
        if lvl is None:
            lvl = level.Global
        report_path = Path(output_path) / f"report_{self.suffix()}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.report(lvl=lvl))

    def save_stats(self, output_path: Path,
                   lvl:Union[level,List[level]]=None) -> None:
        """Sauvegarde l'historique des calculss et les stats."""
        output_path = Path(output_path)
        self.hist_df.to_csv(output_path / f"history_{self.suffix()}.csv", index=False)
        if lvl is None:
            lvl=seqlevel
        if isinstance(lvl,level):
            lvl=[lvl]
        for l in lvl:
            self.stats_group_by(l).to_csv(output_path / f"stat_by_{l}_{self.suffix()}.csv", index=False)

    def save_figures(self , output_dir: Path,
                     lvl:Union[level,List[level]]=None) -> None:
        """
        Sauvegarde les figures dans un dossier.
        """
        output_path = Path(output_dir)
        if lvl is None:
            lvl = [l for l in seqlevel if l!=level.Global]
        for by in lvl:
            for stat in ["mean_accept", "mean_p_value"]:
                self.barplot_by(by=by, stat=stat,
                                show=False, save_path=output_path/f"bar_{stat}_by_{by}_{self.suffix()}.png")
            self.boxplot_by(by=by,show=False,
                            save_path=output_path/f"boxplot_p_value_by_{by.value}_{self.suffix()}.png")

        for metric in ["accept","p_value"]:
            self.heatmap_by(metric=metric,show=False,
                            save_path=output_path/f"heatmap_{metric}_mean_{self.suffix()}.png")

class SequenceAnalyzer(Analyzer):
    """ Classe  d'analyse pour une séquence hérite de ``Analyzer``"""

    _increment = 0  # variable de classe pour compter les instances

    def __init__(self,
                 sequence: NDArray,
                 tests: Optional[List[Tests]] = None,
                 granularities: Optional[List[Union[int, float]]] = None,
                 alpha: float = 0.05,
                 name: str = None,):
        super().__init__(tests=tests, granularities=granularities, alpha=alpha)
        self.sequence = sequence

        if name is None:
            self.name = f"seq_{self.__class__._increment}"
            SequenceAnalyzer._increment += 1
        else:
            self.name = name

    def suffix(self):
        return f"{self.name}(alpha={self.alpha:.1%})"

    def __str__(self) -> str:
        """Représentation string de l'analyse"""
        return f"SequenceAnalyzer_{self.suffix()}"

    def run(self):
        if self.runed:
            raise RuntimeError("les calculs sont déjà excécutés , créer une autre instance")
        print(f"start computing {self}")
        res = analyse_sequence(
            sequence=self.sequence,
            tests=self.tests,
            alpha=self.alpha,
            granularities=self.granularities
        )
        print("end computing")
        self.runed = True
        self.hist_df = res.hist_df

    @property
    def sizes(self)->List[int]:
        """liste des tailles de séquences utilisée lors de  l'évaluation """
        return to_wind_size(granularities=self.granularities ,
                            len_seq=len(self.sequence))

class GeneratorAnalyzer(Analyzer):
    """ Classe d'analyse pour un générateur hérite de ``Analyzer``"""

    def __init__(self,
                 generator: Generators,
                 tests: Optional[List[Tests]] = None,
                 granularities: Optional[List[Union[int, float]]] = None,
                 alpha: float = 0.05,
                 n_repeat: int = 200,
                 seq_len: int = 10000):
        super().__init__(tests=tests, granularities=granularities, alpha=alpha)
        self.generator = generator
        self.n_repeat = n_repeat
        self.seq_len = seq_len

    def suffix(self)->str:
        return f"{str(self.generator)}(alpha={self.alpha:.0%}_k={self.n_repeat}_n={self.seq_len})"

    def __str__(self):
        return f"GeneratorAnalyzer_{self.suffix()}"

    def __repr__(self):
        return str(self)
    def run(self):
        if self.runed:
            raise RuntimeError("les calculs sont déjà excécutés , créer une autre instance")
        print(f"start computing {self}")
        res = evaluate_generator(
                generator=self.generator,
                tests=self.tests,
                alpha=self.alpha,
                n_repeat=self.n_repeat,
                seq_len=self.seq_len,
                granularities=self.granularities
        )
        self.runed=True
        self.hist_df = res.hist_df

    @property
    def sizes(self)->List[int]:
        """liste des tailles de séquences utilisée lors de  l'évaluation """
        return to_wind_size(granularities=self.granularities,len_seq=self.seq_len)

    @classmethod
    def compare_multiple(cls,generators:List[Generators],**kwargs)->Dict[str,'GeneratorAnalyzer']:
        """compare plusieurs générateurs"""
        results = {}
        for gen in generators:
            analyzer = cls(gen)