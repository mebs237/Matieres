from src.models.system import RangelandSystem
from src.visualization.plotter import RangelandPlotter
from src.utils.numerical import solve_system
import numpy as np

def main():
    # Création du système
    system = RangelandSystem()

    # Analyse des trajectoires
    trajectory_plotter = TrajectoryPlotter(system)
    phase_plotter = PhasePlotter(system)

    # Conditions initiales
    initial_state = np.array([0.2, 0.3])

    # Résolution
    result = solve_system(
        func=system.equation,
        initial_state=initial_state,
        t_span=(0, 100),
        E=0.1
    )

    # Visualisation
    if result['success']:
        plotter = RangelandPlotter(system)
        plotter.plot_trajectory([initial_state], E=0.1)

if __name__ == "__main__":
    main()