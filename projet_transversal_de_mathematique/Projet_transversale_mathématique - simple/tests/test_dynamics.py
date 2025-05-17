from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals

@dataclass
class RangelandSystem:
    """Système complet de dynamique des pâturages"""

    # Paramètres du système
    Rg: float = 0.27  # Taux de croissance des herbes vivaces
    Rw: float = 0.4   # Taux de croissance des mauvaises herbes
    a: float = 0.31   # Paramètre de compétition a
    b: float = 0.6    # Paramètre de compétition b
    c: float = 1.07   # Paramètre de compétition c

    # Valeurs critiques de E (calculées une fois)
    E1: float = 0.3199208673727
    E2: float = 40.6057908272657

    def equation(self, t: float, state: np.ndarray, E: float) -> np.ndarray:
        """Équations différentielles du système"""
        g, w = state
        # Votre implémentation actuelle...

    def find_equilibria(self, E: float):
        """Trouve les points d'équilibre"""
        # Votre fonction equilibres()...

    def classify_equilibrium(self, E: float):
        """Classifie les points d'équilibre"""
        # Votre fonction class_equilibrium()...

    def compute_separatrix(self, E: float):
        """Calcule la séparatrice"""
        # Votre fonction separatrix_points()...