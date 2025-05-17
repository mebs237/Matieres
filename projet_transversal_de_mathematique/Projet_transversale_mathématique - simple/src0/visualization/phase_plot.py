from ..models.equilibrium import RangelandSystem
import matplotlib.pyplot as plt
import numpy as np

class PhasePlotter:
    """Visualisation du portrait de phase"""

    def __init__(self, system: RangelandSystem):
        self.system = system

    def plot_separatrix(self, E: float):
        """Ancien 'plot_separatrix' de Equilibres.py"""

    def plot_equilibria(self, E: float):
        """Ancien code de plot_eq.py"""