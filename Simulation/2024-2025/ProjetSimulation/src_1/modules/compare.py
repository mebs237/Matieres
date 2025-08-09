"""Module pour la comparaison des générateurs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .Tests import Tests
from .Generators import Generators
from .Analysing import GeneratorAnalyzer, level
from .Utils import requires__run

@dataclass
class ComparatorGenerator:
    """Classe pour comparer plusieurs générateurs de nombres aléatoires."""
    generators: List[Generators]
    tests: Optional[List[Tests]] = None
    granularities: Optional[List[Union[int, float]]] = None
    alpha: float = 0.05
    n_repeat: int = 200
    seq_len: int = 10_000
    runed: bool = False
    results: Dict[str, GeneratorAnalyzer] = field(init=False, default_factory=dict)

    def run(self) -> None:
        """Évalue tous les générateurs."""
        if self.runed:
            raise RuntimeError("Les calculs sont déjà exécutés, créer une autre instance")

        print("Début de l'évaluation des générateurs...")
        for gen in self.generators:
            name = str(gen)
            print(f"Évaluation de {name}...")
            analyzer = GeneratorAnalyzer(
                generator=gen,
                tests=self.tests,
                granularities=self.granularities,
                alpha=self.alpha,
                n_repeat=self.n_repeat,
                seq_len=self.seq_len
            )
            analyzer.run()
            self.results[name] = analyzer
        self.runed = True
        print("Tous les générateurs ont été évalués.")

    @requires__run
    def to_stats_table(self) -> pd.DataFrame:
        """Retourne un tableau global des statistiques."""
        data = {name: analyzer.stats_group_by(level.Global).iloc[0]
                for name, analyzer in self.results.items()}
        return pd.DataFrame(data).round(4)

    @requires__run
    def to_aggregated_table(self, by: level = level.Bytest) -> pd.DataFrame:
        """Retourne un tableau agrégé par niveau."""
        all_tables = {}
        for name, analyzer in self.results.items():
            df = analyzer.stats_group_by(by)
            df.columns = pd.MultiIndex.from_product([[name], df.columns])
            all_tables[name] = df
        return pd.concat(all_tables.values(), axis=1)

    @requires__run
    def compare_barplot(self,
                       by: level = level.Bytest,
                       stat: Literal["mean_p_value","mean_accept"] = "mean_p_value",
                       show: bool = True,
                       save_path: Optional[Path] = None):
        """Compare les statistiques avec un barplot."""
        df = self.to_aggregated_table(by)
        df = df.stack(level=0).reset_index()
        df = df.rename(columns={'level_1': 'Generator'})

        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x=by.value, y=stat, hue='Generator')
        plt.title(f"Comparaison de {stat} par {by.value}")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(title="Générateur", bbox_to_anchor=(1.05, 1))
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    @requires__run
    def compare_boxplot(self,
                       by: level = level.Bytest,
                       metric: Literal["p_value","accept"] = "p_value",
                       show: bool = True,
                       save_path: Optional[Path] = None):
        """Compare les distributions avec un boxplot."""
        all_dfs = []
        for name, analyzer in self.results.items():
            df = analyzer.hist_df.copy()
            df['Generator'] = name
            all_dfs.append(df)

        df = pd.concat(all_dfs, ignore_index=True)
        df.dropna()
        df['accept'] = df['accept'].astype(int)
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=by.value, y=metric, hue='Generator')
        plt.title(f"Distribution de {metric} par {by.value}")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(title="Générateur", bbox_to_anchor=(1.05, 1))
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    @requires__run
    def report(self, output_path: Path) -> None:
        """Génère un rapport comparatif."""
        output_path.mkdir(exist_ok=True)

        # Statistiques globales
        stats_df = self.to_stats_table()

        report = ["# Rapport comparatif des générateurs\n"]
        report.append("## Statistiques globales\n")
        report.append(stats_df.to_markdown() + "\n\n")

        # Ajout des graphiques et interprétations par niveau
        for by in [level.Bytest, level.Bysize]:
            report.append(f"\n## Analyse par {by.value}\n")

            # Sauvegarde et référence des graphiques
            barplot_path = output_path / f"barplot_{by.value}.png"
            boxplot_path = output_path / f"boxplot_{by.value}.png"

            self.compare_barplot(by=by, show=False, save_path=barplot_path)
            self.compare_boxplot(by=by, show=False, save_path=boxplot_path)

            report.append(f"\n![Comparaison par {by.value}]({barplot_path.name})")
            report.append(f"\n![Distribution par {by.value}]({boxplot_path.name})\n")

        with open(output_path / "rapport_comparatif.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    @requires__run
    def save_all(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Sauvegarde tous les résultats."""
        # Création du dossier de sortie
        output_dir = f"Results_compare_a={self.alpha:.1%}_k={self.n_repeat}_n={self.seq_len}" if output_dir is None else output_dir
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Génération du rapport
        self.report(output_path)

        # Sauvegarde des tableaux de statistiques
        self.to_stats_table().to_csv(output_path / "statistiques_globales.csv")

        # Sauvegarde des tableaux agrégés
        for by in [level.Bytest, level.Bysize]:
            self.to_aggregated_table(by).to_csv(output_path / f"statistiques_par_{by.value}.csv")

        # Sauvegarde des graphiques comparatifs
        for by in [level.Bytest, level.Bysize]:
            self.compare_barplot(
                by=by,
                show=False,
                save_path=output_path / f"barplot_{by.value}.png"
            )
            self.compare_boxplot(
                by=by,
                show=False,
                save_path=output_path / f"boxplot_{by.value}.png"
            )