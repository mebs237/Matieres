from src.rangeland import RangelandSystem
from src.visualizer import RangelandVisualizer

# Créer une instance du système
system = RangelandSystem()

# Analyser les équilibres
equilibria = system.find_equilibria(E=0.1)

# Visualiser les résultats
viz = RangelandVisualizer(system)
viz.plot_phase_portrait(E=0.1)