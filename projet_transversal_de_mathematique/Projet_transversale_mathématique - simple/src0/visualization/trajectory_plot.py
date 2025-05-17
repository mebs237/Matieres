import matplotlib.pyplot as plt
import numpy as np
from ..models.equilibrium import RangelandSystem
from ..utilsclass.numerical_tools import solve_system

class RangelandPlotter:
    def __init__(self, system: RangelandSystem):
        self.system = system

    def plot_trajectory(self, initial_conditions: list, E: float, t_max: float = 100):
        """Trace les trajectoires du système."""
        plt.figure(figsize=(10, 8))

        # Implémentation du tracé...

        plt.xlabel('g (herbes vivaces)')
        plt.ylabel('w (mauvaises herbes)')
        plt.title(f'Trajectoires pour E = {E}')
        plt.grid(True)
        plt.show()


class TrajectoryPlotter:
    """Fusion des fonctionnalités de trajectoires.py et trajectoires2.py"""

    def __init__(self, system: RangelandSystem):
        self.system = system

    def plot_trajectory(self, initial_conditions: list, E: float):
        """Ancien code de trajectoires.py"""
        n_ic_side = 5
        initial_conditions = []

        for i in range(n_ic_side):
            w0 = np.random.uniform(0, 1)
            initial_conditions.append(np.array([0,w0]))
            initial_conditions.append(np.array([1,w0]))
        for i in range(n_ic_side):
            g0 = np.random.uniform(0, 1)
            initial_conditions.append(np.array([g0,0]))
            initial_conditions.append(np.array([g0,1]))



        # Résolution des équations différentielles
        trajectories = []
        t_span = (0, 100) # periode de simulation
        t_eval = np.linspace(t_span[0] , t_span[1] , 1000) # Période de simulation
        for ic in initial_conditions:
            sol = solve_system(self.equation, t_span, ic, method='RK45', max_step=0.1,args=(E))
            trajectories.append(sol.y)

        # Tracé des trajectoires avec flèches
        plt.figure(figsize=(8, 6))
        for traj in trajectories:
            g_vals = traj[0]
            w_vals = traj[1]
            plt.plot(g_vals,w_vals,color='red')
            # Ajout de flèches toutes les 100 points
            for i in range(0, len(g_vals)-1, 200):
                dx = g_vals[i+1] - g_vals[i]
                dy = w_vals[i+1] - w_vals[i]
                plt.arrow(g_vals[i], w_vals[i], dx, dy ,
                lw=0, length_includes_head=False, head_width=0.02 , color='red')

        # Marquage des équilibres triviaux
        equilibres = [(0, 0), (1, 0), (0, 1)]
        for eq in equilibres:
            plt.plot(eq[0], eq[1],'bo', markersize=16)

        # Configuration du graphique
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('g')
        plt.ylabel('w')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title(f'Trajectoires pour E={E}')
        plt.show()

    def plot_trajectory_with_arrows(self, initial_conditions: list, E: float):
        """Ancien code de trajectoires2.py"""