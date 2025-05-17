import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
from rangeland import RangelandSystem , DefaultSystem , EquilibriumType


@dataclass
class RangelandVisualizer:
    """Classe pour la visualisation du système Rangeland avec superposition aisée"""

    system: 'RangelandSystem' = field(default_factory= DefaultSystem())
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100

    def _create_figure_if_needed(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Crée une nouvelle figure si aucun axes n'est fourni"""
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return ax

    def _setup_axes(self, ax: plt.Axes) -> None:
        """Configuration de base des axes"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_xlabel('Grass (g)')
        ax.set_ylabel('Weed (w)')

    def plot_equilibria(self, *args, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Trace uniquement les points d'équilibre sans autres éléments

        Args:
            ax: Axes matplotlib existants (optionnel)
        """
        ax = self._create_figure_if_needed(ax)
        equilibria = self.system.get_classified_equilibria(*args)

        for eq in equilibria:
            ax.plot(*eq.coordinates, 'o', markersize=10,
                   label=f'Equilibrium ({eq.nature.value})')

        self._setup_axes(ax)
        ax.legend()
        return ax

    def distinguish_equilibria(self, *args, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Trace les équilibres avec un code couleur standard

        Args:
            ax: Axes matplotlib existants (optionnel)
        """
        ax = self._create_figure_if_needed(ax)
        equilibria = self.system.get_classified_equilibria(*args)

        for eq in equilibria:
            color = {
                EquilibriumType.STABLE: 'green',
                EquilibriumType.UNSTABLE: 'red',
                EquilibriumType.SADDLE: 'yellow'
            }.get(eq.nature, 'black')

            ax.plot(*eq.coordinates, 'o', markersize=12,
                   markeredgecolor='black', markeredgewidth=1,
                   label=f'{eq.nature.value.capitalize()}',
                   color=color)

        self._setup_axes(ax)
        ax.legend()
        return ax

    def plot_vector_field(self, *args, grid_size: int = 20,
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Trace le champ de vecteurs

        Args:
            grid_size: Densité de la grille
            ax: Axes matplotlib existants (optionnel)
        """
        ax = self._create_figure_if_needed(ax)

        def F(u):
            return self.system.equation(0, u, *args)

        X, Y = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        U, V = F([X, Y])

        ax.quiver(X, Y, U, V, color='blue', alpha=0.6)
        self._setup_axes(ax)
        ax.set_title('Vector field')
        return ax

    def plot_separatrix(self, *args, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Trace la séparatrice

        Args:
            ax: Axes matplotlib existants (optionnel)
        """
        ax = self._create_figure_if_needed(ax)
        w, g = self.system.separatrix_points(*args)

        if len(w) > 0:
            ax.plot(g, w, '--', color='purple', linewidth=2,
                   label='Separatrix', alpha=0.8)

        self._setup_axes(ax)
        ax.legend()
        return ax

    def plot_trajectories(self,
                          initial_conditions: List[Tuple[float, float]],
                          *args, t_span: Tuple[float, float] = (0, 100),
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Trace des trajectoires à partir de conditions initiales

        Args:
            initial_conditions: Liste de tuples (g0, w0)
            t_span: Intervalle de temps
            ax: Axes matplotlib existants (optionnel)
        """
        ax = self._create_figure_if_needed(ax)

        for g0, w0 in initial_conditions:
            sol = solve_ivp(
                lambda t, u: self.system.equation(t, u, *args),
                t_span,
                [g0, w0],
                method='RK45'
            )
            ax.plot(sol.y[0], sol.y[1], '-', linewidth=2,
                   label=f'Traj. from ({g0:.1f}, {w0:.1f})')

        self._setup_axes(ax)
        ax.legend()
        return ax

    def plot_phase_portrait(self, *args, show: bool = True,
                           save: Optional[str] = None) -> None:
        """
        Portrait de phase complet avec superposition de tous les éléments

        Args:
            show: Afficher le graphique
            save: Chemin pour sauvegarde (optionnel)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Superposition ordonnée
        self.plot_vector_field(*args, ax=ax)
        self.plot_separatrix(*args, ax=ax)
        self.distinguish_equilibria(*args, ax=ax)

        # Configuration finale
        ax.set_title(f'Phase portrait for {args if args else "default parameters"}')
        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


sys = DefaultSystem()
print(sys.equation(0 , np.array([0,0]) , 5))
print(sys.params)
print(sys.equation)