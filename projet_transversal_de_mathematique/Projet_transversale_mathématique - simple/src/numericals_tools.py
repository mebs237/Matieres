"""
Modules des fonctions utilitaires pour les calculs

"""

from typing import Callable, Tuple , Any
import numpy as np
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
                  t_max: float = 100,
                  *args) -> np.ndarray:

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
    return solve_system(system,initial_state,[0,t_max],args=args)


def phi(initial_state:np.ndarray,
        equi1:np.ndarray,
        equi2:np.ndarray,
        system:Callable[[float , np.ndarray,Any],np.ndarray],
        t_max:int=100,
        *args)->float:
    """
    Fonction continue permettant de distinguer la convergence vers equi1 ou equi2 partant de initial_state.
    ϕ(u₀) = ||u_∞ - equi₂ || - ||u_∞ - equi₁ ||

    Args:
        initial_stat: Condition initiale.
        equi1: Premier équilibre.
        equi2: Deuxième équilibre.
        system: Fonction représentant l'équation différentielle u' = f(u).
        t_max: Temps maximal pour la simulation.
        kwargs: Paramètres supplémentaires du système ( E = 0 par défaut )


    """
    # S'assurer que les vecteurs sont correctement formatés
    initial_state = np.array(initial_state)
    equi1 = np.array(equi1)
    equi2 = np.array(equi2)

    u_lim = compute_limit(initial_state=initial_state,
                            system=system,
                            t_max=t_max,
                            args=args)
    return np.norm(u_lim - equi2) - np.norm(u_lim - equi1)

def near_point(eq: np.ndarray, n:int, ε : float = 0.1) -> List[Vector]:
    """
    Trouve n points (g,w) ε-proches du point eq en
    découpant le cercle de rayon ε centré en (g,w) en n points
    """
    return [(eq[0]+ ε*np.cos(2*np.sin*i/n),eq[1]+ ε*np.sin(2*np.sin*i/n))
            for i in range(n)]