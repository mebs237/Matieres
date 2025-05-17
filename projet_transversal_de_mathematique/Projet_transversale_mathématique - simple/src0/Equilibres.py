from typing import (Callable  , Union , Tuple , List)
from numpy import (array  , sqrt , float64, float16 , float32 , linspace)
from numpy.typing import NDArray
from scipy.differentiate import jacobian
from scipy.linalg import eigvals , norm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np

Num = Union[float,float64, float16 , float32 ]
vector = NDArray[Num]

# déclaration des constantes du projet
a , b , c = 0.31 , 0.6 , 1.07
Rg , Rw = 0.27 , 0.4

#delta du polynôme en E
D = 16*((b*c -1)**2)*(a-b)*(a-1)*b
# coefficients du polynôme en E
a2 , b2 , c2 = ((1-b*c)*a)**2 , -2*(b*c-1)*(a*(b+1)-2*b) , (1-b)**2
# Condition sur E pour d'autre equilibres triviaux
#E1 , E2 = (-b2-sqrt(D))/(2*a2) , (-b2+sqrt(D))/(2*a2)
E1 , E2 = 0.3199208673727 , 40.6057908272657


Es = [ 0, 0.1, 0.3, 0.4 , 40.6 , 41]


def equation(t, v, E=0.00):
    """
    Membre de droite de l'EDO du rangeland.

    Args:
        t: Temps (non utilisé mais requis par solve_ivp)
        v: Vecteur d'état [g, w] defni dans le projet
        E: rapport de proportionnalité avec le bétail

    Returns:
        Vecteur des dérivées [dg/dt, dw/dt]
    """
    g, w = v

    if E == 0 and g == 0:
        dgdt = Rg*g*(1-g- 0.6*w)
        dwdt = Rw*w*(1-w-1.07*g)
    else:
        dgdt = Rg*g*(1-g- 0.6*w*((E + g)/(0.31*E + g)))
        dwdt = Rw*w*(1-w-1.07*g*((0.31*E + g)/(E + g)))
    return array([dgdt , dwdt])

def Pg(x:float, E:float) -> float:
    """
    polynôme g dont une racine g* forme un équilibre avec un w* correspondant
    """
    if x == 0 and E == 0:
        return 0.4
    return 0.358 * x**2 + (0.11098 * E - 0.4) * x + 0.29 * E




def equilibres(E:float)-> List[Tuple]:
    """
    Prenant un E en entrée, retourne les équilibres du système
    """

    # les équilibres triviaux
    equilibrium = [(0, 0), (1, 0), (0, 1)]
    # coefficients du polynômes en g
    a1 , b1 , c1 = 0.358 , 0.1109*E - 0.4 , 0.29*E
    delta = b1**2 -4*a1*c1
    if E1 < E < E2: # pas d'équilibres non triviaux
        return equilibrium

    elif E==E1 or E==E2: # un seul équilibre non trivial
        g = -b1/(2 * a1)
        w = dubble_u(g , E)
        #if 0 < g < 1 and 0 < w1 < 1:
        equilibrium.append((g, w))
    else  :     # deux équilibres non triviaux
        g1 = (-b1 - sqrt(delta)) / (2 * a1)
        g2 = -b1/a1 - g1
        w1 , w2 = dubble_u(g1,E) , dubble_u(g2,E)

        #if 0 < g1 < 1 and 0 < w1 < 1:
        equilibrium.append((g1, w1))
        #if 0 < g2 < 1 and 0 < w2 < 1:
        equilibrium.append((g2, w2))

    return equilibrium

def class_equilibrium(E:float)-> dict[Tuple, str]:
    """
    Identifie la nature des équilibres du système en fonction de la valeur de E.

    Returns:
        Dictionnaire associant chaque équilibre à sa classification
              ('stable', 'instable' ou 'selle')
    """
    eqs = equilibres(E)
    class_eq = {}
    for eq in eqs:
        J = Jac(eq , E)
        # Valeurs propres de la jacobienne en eq
        eigenvalues = eigvals(J)
        # Partie réelle des valeurs propres
        real_part = array([t.real for t in eigenvalues])
        # Les valeurs propres sont toutes de partie réelle négative
        if all(real_part<0):
            class_eq[tuple(eq)] = 'stable'
        # Les valeurs propres sont toutes de partie réelle positive
        elif all(real_part>0):
            class_eq[tuple(eq)] = 'instable'
        else:
            class_eq[tuple(eq)] = 'selle'

    return class_eq


def class_equilibrium2(E:float)-> dict[Tuple, str]:
    """
    Version alternative de class_equilibirum utilisant la fonction jacobian de scipy.

    """
    eqs = equilibres(E)
    class_eq = {}
    for eq in eqs:
        jac = jacobian(lambda v : equation(0, v, E), eq) # On utilise la fonction jacobian de scipy pour calculer la matrice jacobienne
        eigenvalues = eigvals(jac)
        # Partie réelle des valeurs propres
        real_part = array([t.real for t in eigenvalues])
        # Les valeurs propres sont toutes de partie réelle négative
        if all(real_part<0):
            class_eq[tuple(eq)] = 'stable'
        # Les valeurs propres sont toutes de partie réelle positive
        elif all(real_part>0):
            class_eq[tuple(eq)] = 'instable'
        else:
            class_eq[tuple(eq)] = 'selle'

    return class_eq



def cvg(u0:vector, u_star:vector ,f:Callable[[vector],vector], t_max=100, tol=1e-6  )->bool:
    """
    Détermine si la solution u de condition initiale u0
    converge vers u_star.

    Args:
        u0: Condition initiale.
        u_star: Équilibre, la racine de f (point fixe).
        f: Fonction représentant l'équation différentielle u' = f(u).
        t_max: Temps maximal de simulation.
        tol: Tolérance pour la convergence.

    Returns:
        bool: True si la solution converge vers u_star, False sinon.
    """
    # Extraire la dernière valeur de u(t)
    u_final = ulim(u0,f,t_max)

    # Vérifier si u_final est proche de u_star
    return norm(u_final - u_star) < tol


def phi(u0:vector, equi1:vector, equi2:vector, f:Callable=equation, t_max:int=100, E:float=0.00)->float:
    """
    Fonction continue permettant de distinguer la convergence vers equi1 ou equi2 partant de u0.
    f(u₀) = ||u_∞ - equi₂ || - ||u_∞ - equi₁ ||

    Args:
        u0: Condition initiale.
        equi1: Premier équilibre.
        equi2: Deuxième équilibre.
        f: Fonction représentant l'équation différentielle u' = f(u).
        t_max: Temps maximal pour la simulation.
        E: Paramètre d'exploitation ou environnemental.

    Returns:
        float:
            - Strictement positif, si convergence vers equi1
            - Strictement négatif, si convergence vers equi2
            - Proche de zéro, si ne converge vers aucun des deux
    """
    # S'assurer que les vecteurs sont correctement formatés
    u0 = array(u0)
    equi1 = array(equi1)
    equi2 = array(equi2)

    u_final = ulim(u0, f, t_max, E)
    return norm(u_final - equi2) - norm(u_final - equi1)


def intercept_g(w:float, E:float, t_max:int=100)->float:
    """
    Calcule le g tel que (g,w) est l'intersection entre la droite y = w et la courbe séparatrice.

    Args:
        w (float): pourcentage de mauvaises herbes
        E (float): proportion de bétail
        t_max (int, optional): Temps maximal pour la simulation. Defaults to 100.

    Returns:
        float: Valeur de g correspondant à l'intersection, ou None si aucune intersection n'existe.
    """
    # Récupération des classes d'équilibres
    classes = class_equilibrium2(E)

    # Récupérer les équilibres stables
    stable = [eq for eq , cls in classes.items() if cls == 'stable']

    nb_stable = len(stable)

    # Un seul équilibre stable : pas de courbe séparatrice
    if nb_stable < 2:
        return None

    # On sait qu'il y a soit 1 ou 2 équilibres stables
    equi1, equi2 = stable

    def objective(g):
        # S'assurer que les vecteurs sont correctement formatés
        u0 = array([g, w])
        equi1_array = array(equi1)
        equi2_array = array(equi2)
        return phi(u0, equi1_array, equi2_array, equation, t_max, E)

    try:
        return brentq(objective, 0, 1)
    except ValueError:
        # Si brentq ne trouve pas de solution, retourner None
        return None


def separatrix_points(E:float,grid_size:int=200):
    """
    Génère les points de la courbe séparatrice aussi finement que souhaité.

    Args:
        E (float): Paramètre d'exploitation ou environnemental
        grid_size (int, optional): Nombre de points à générer. Defaults to 200.

    Returns:
        tuple: (w, g) où w et g sont des tableaux NumPy contenant les coordonnées des points de la courbe séparatrice. Si aucune courbe séparatrice n'existe, retourne des tableaux vides.
    """
    # On se fixe un ensemble de w entre 0 et 1
    w_grid = linspace(0,1,grid_size)

    # On trouve le g* de l'intersection entre la courbe et y = w
    g_separatrix = [intercept_g(w,E,t_max=500) for w in w_grid]

    # Filtrer les valeurs None
    valid_indices = [i for i, g in enumerate(g_separatrix) if g is not None]

    if not valid_indices:
        # Aucune courbe séparatrice n'existe
        return np.array([]), np.array([])

    # Extraire les valeurs valides
    w_valid = w_grid[valid_indices]
    g_valid = np.array([g_separatrix[i] for i in valid_indices])

    return w_valid, g_valid



