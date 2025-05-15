"""
Modules des fonctions utilitaires pour les calculs

"""

from typing import Callable, Tuple , Any , List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp , OdeSolution
import matplotlib.pyplot as plt

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
    return solve_system(system,initial_state,[0,t_max],*args)


def phi(initial_state:np.ndarray,
        eq1:np.ndarray,
        eq2:np.ndarray,
        system:Callable[[float , np.ndarray,Any],np.ndarray],
        t_max:int=100,
        *args)->float:
    """
    Fonction continue qui determine vers quel equilibre eq1 ou eq2 tend la solution
    ϕ(u₀) = ||u_∞ - equi₂ || - ||u_∞ - equi₁ ||

    Args:
        initial_stat: Condition initiale.
        eq1: Premier équilibre.
        eq2: Deuxième équilibre.
        system: Fonction représentant l'équation différentielle u' = f(u).
        t_max: Temps maximal pour la simulation.
        args: Paramètres supplémentaires pour E ( E = 0 par défaut )

    Returns:
    float: Distance signée indiquant vers quel équilibre la solution tend
    """
    # S'assurer que les vecteurs sont correctement formatés
    initial_state = np.array(initial_state)
    eq1 = np.array(eq1)
    eq2 = np.array(eq2)

    u_lim = compute_limit(initial_state=initial_state,
                            system=system,
                            t_max=t_max,
                            *args)
    return np.norm(u_lim - eq2) - np.norm(u_lim - eq1)

def near_point(eq: np.ndarray, n:int, eps : float = 0.1) -> List[NDArray[np.float64]]:
    """
    Trouve n points (g,w) ε-proches du point eq en
    découpant le cercle de rayon ε centré en (g,w) en n points
    """
    return [(eq[0]+ eps*np.cos(2*np.pi*i/n),
             eq[1]+ eps*np.sin(2*np.pi*i/n))  for i in range(n)]




def plot_vector_field_near_equilibria(func, equilibria, eps, n_points:int=20)->None:
    """
    Trace les vecteurs du champ func(u) autour des points d'équilibre dans un cercle de rayon eps.

    Arguments:
    func -- fonction de champ de vecteurs, prend un tableau de coordonnées et retourne les vecteurs
    equilibria -- liste des points d'équilibre (coordonnées)
    eps -- rayon autour des équilibres
    n_points -- nombre de points à générer sur le cercle (par défaut 20)
    """
    fig , ax = plt.subplots(figsize=(8, 6))

    for eq in equilibria:
        points = near_point(np.array(eq), n_points, eps)
        X, Y = zip(*points)
        U, V = func([np.array(X), np.array(Y)])

        plt.quiver(X, Y, U, V, color='blue', alpha=0.6)
        plt.plot(eq[0], eq[1], 'ro')  # Marquer le point d'équilibre en rouge
        circle = plt.Circle((eq[0], eq[1]), eps, color='red', fill=False, linestyle='--')
        ax.add_artist(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def F(u):
    return -u[0]**2 + 2*u[1] , 2*u[1]-5


equilibria = [
    np.array([-5 , 5/2]) ,
    np.array([5 , 5/2])
]

plot_vector_field_near_equilibria(func=F,eps=0.5,equilibria=equilibria)