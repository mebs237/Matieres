"""
Modules des fonctions utilitaires pour les calculs

"""

from typing import Callable, Tuple , Any , List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp , OdeSolution


def solve_system(system: Callable,
                 initial_state: np.ndarray,
                 t_span: Tuple[float, float],
                 *args )->OdeSolution:
    """resous le système numériquement"""
    try :
        sol = solve_ivp(system, t_span=t_span, y0=initial_state, method='RK45', rtol=1e-8, atol=1e-10 , args = args)

        return sol
    except ValueError as e:
        print(f"Erreur de valeur lors de l'intégration: {e}")
    except RuntimeError as e:
        print(f"Erreur d'exécution lors de l'intégration: {e}")
        return None

def compute_limit(system:Callable[[float,np.ndarray],np.ndarray],
                  initial_state:np.ndarray,
                  *args,
                  t_max: float = 100,
                  ) -> np.ndarray:

    """
    Calcule la limite d'une solution du système différentiel

    Args:
        initial_state: Condition initiale de la solution.
        system: Fonction définissant l'équation différentielle.
        t_max: Temps maximal pour la simulation (pseudo-infini).
        **kwargs: paramètres supplémentaire (E = 0 par défaut)
    """
    # S'assurer que u0 est un vecteur 1-dimensionnel
    initial_state = np.array(initial_state)
    return solve_system(system,initial_state,[0,t_max],*args)


def phi(initial_state:np.ndarray,
        eq1:np.ndarray,
        eq2:np.ndarray,
        system:Callable[[float , np.ndarray,Any],np.ndarray],
        *args,
        t_max:int=100
        )->float:
    """
    Distance signée indiquant vers quel équilibre la solution partant ``initial_state``
    ϕ(u₀) = ||u_∞ - equi₂ || - ||u_∞ - equi₁ ||

    Args:
        initial_stat: Condition initiale.
        eq1: Premier équilibre.
        eq2: Deuxième équilibre.
        system: Fonction représentant l'équation différentielle u' = f(u).
        t_max: Temps maximal pour la l'intégration.
        args: Paramètres supplémentaires pour ( typiquement E  par défaut )
    """
    # S'assurer que les vecteurs sont correctement formatés
    initial_state = np.array(initial_state)
    eq1 = np.array(eq1)
    eq2 = np.array(eq2)

    u_lim = compute_limit(initial_state=initial_state,
                            system=system,
                            t_max=t_max,
                            *args)
    return np.linalg.norm(u_lim - eq2) - np.linalg.norm(u_lim - eq1)

def near_point(eq: np.ndarray, n:int, eps : float = 0.1) -> List[NDArray[np.float64]]:
    """
    Trouve n points (g,w) ε-proches du point eq en
    découpant le cercle de rayon ε centré en (g,w) en n points
    """
    return [(eq[0]+ eps*np.cos(2*np.pi*i/n),
             eq[1]+ eps*np.sin(2*np.pi*i/n))  for i in range(n)]



