from typing import Tuple, Optional
import numpy as np

class DynamicsAnalyzer:
    """Analyse des dynamiques du système."""

    def dubble_u(self, g: float, E: float) -> float:
        """Ancien 'dubble_u' de Equilibres.py"""
    def dubble_u(g:float , E:float)->float:
        """
        retourne la valeur de w en fonction de g tel que g , f(g) est un équilibre
        """
        if g == 0 and E == 0:
            return 1
        return 1 - 1.07 * g * (0.31 * E + g)/(E + g)


    def Pg(self, x: float, E: float) -> float:
        """Ancien 'Pg' de Equilibres.py"""

    def delta_e(self, E: float) -> float:
        """Ancien 'delta_e' de Equilibres.py"""