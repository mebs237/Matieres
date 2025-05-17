from src.rangeland import RangelandSystem , DefaultSystem
from src.visualizer import RangelandVisualizer

# Créer une instance du système
system = DefaultSystem

# Analyser les équilibres
equilibria = system.find_equilibria(E=0.3)

# Visualiser les résultats
viz = RangelandVisualizer(system)
viz.plot_phase_portrait()