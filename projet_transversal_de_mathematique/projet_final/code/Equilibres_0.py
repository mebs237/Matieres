from typing import (Callable  , Union , Tuple , List)
from numpy import (array  , sqrt , float64, float16 , float32 , linspace)
from numpy.typing import NDArray
from scipy.differentiate import jacobian
from scipy.linalg import eigvals , norm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np

Rg , Rw = 0.27 , 0.4
Num = Union[float,float64, float16 , float32 ]
vector = NDArray[Num]

Es = array([
        0,
        0.1,
        0.3,
        0.4
          ])

def equation(t, v, E=0.00):
    """
    Membre de droite de l'EDO du rangeland.

    Args:
        t: Temps (non utilisé mais requis par solve_ivp)
        v: Vecteur d'état [g, w] où g est la biomasse d'herbe et w est la biomasse ligneuse
        E: Paramètre d'exploitation ou environnemental

    Returns:
        Vecteur des dérivées [dg/dt, dw/dt]
    """
    g, w = v[0], v[1]

    if E == 0 and g == 0:
        return array([0.27*g*(1-g- 0.6*w),
                     0.4*w*(1-w-1.07*g)])
    else:
        return array([Rg*g*(1-g- 0.6*w*((E + g)/(0.31*E + g))),
                     Rw*w*(1-w-1.07*g*((0.31*E + g)/(E + g)))
                   ])

def Pg(x:float, E:float) -> float:
    """
    Permet de représenter la fonction polynôme g avec comme paramètre E
    """
    if x == 0 and E == 0:
        return 0.4
    return 0.358 * x**2 + (0.11098 * E - 0.4) * x + 0.29 * E

def delta_root_finder() -> vector:
    """
    Retourne les racines du delta du polynôme en g
    """
    return brentq(delta_e, 0, 2), brentq(delta_e, 30, 120)

def delta_e(E:float) -> float:
    """
    Retourne la valeur en un point du delta du polynôme en g
    (le delta du polynôme en g est une fonction du second degré en E)
    """
    return (0.11098 * E - 0.4)**2 - 4 * 0.358 * 0.29 * E


def dubble_u(g:float , E:float)->float:
    """
    retourne la valeur de w en fonction de la valeur de g correspondante et E (E)
    """
    if g == 0 and E == 0:
        return 1
    return 1 - 1.07 * g * (0.31 * E + g)/(E + g)

def equilibres(E:float)-> List[Tuple]:
    """
    Prenant un E en entrée, retourne les équilibres du système
    """
    #coefficients du polynôme en g
    a , b , c = 0.358 ,  0.11098 * E - 0.4 ,  0.29 * E

    # les équilibres triviaux
    equilibrium = [(0, 0), (1, 0), (0, 1)]

    delta = b**2 - 4 * a * c
    if delta > 0:
        g1 = (-b - sqrt(delta)) / (2 * a)
        g2 = -b/a - g1
        w1 , w2 = dubble_u(g1,E) , dubble_u(g2,E)

        if 0 < g1 < 1 and 0 < w1 < 1:
            equilibrium.append((g1, w1))
        # on vérifie que les valeurs sont bien dans l'intervalle [0,1]
        if 0 < g2 < 1 and 0 < w2 < 1:
            equilibrium.append((g2, w2))
    elif delta == 0:
        g = -b/(2 * a)
        w1 = dubble_u(g , E)
        if 0 < g < 1 and 0 < w1 < 1:
            equilibrium.append((g, w1))
    return equilibrium


def Jac(u:vector,E:float)->vector:
    """
        matrice Jacobienne du membre de droite de l'équation différentielle
    """
    g , w = u[0] , u[1]
    if g == 0 and E == 0:
        df1_dg = 0.27*(1-0.6*w)
        df2_dg = -0.4*w*(1.07)
        df1_dw = 0.27*g*(-0.6)
        df2_dw = 0.4*(1-w)
    else:
        df1_dg = 0.27*(1-g-0.6*w*(E+g)/(0.31*E+g)+g*(-1-0.6*w*(-0.69*E)/(0.31*E+g)**2))
        df2_dg = 0.4* w*(-1.07*(0.31*E+g)/(E+g)-1.07*g*(0.69*E)/(E+g)**2)
        df1_dw = 0.27*g*(-0.6*(E+g)/(0.31*E+g))
        df2_dw =0.4*(1-w-1.07*g*(0.31*E+g)/(E+g)-w)

    return array([[ df1_dg , df1_dw ],
                  [ df2_dg , df2_dw ]
                ])

def class_equilibrium(E:float)-> dict[Tuple, str]:
    """
    Identifie la nature des équilibres du système en fonction de la valeur de E.

    Args:
        E: Paramètre d'exploitation ou environnemental

    Returns:
        dict: Dictionnaire associant chaque équilibre à sa classification
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


def class_equilibriumv2(E:float)-> dict[Tuple, str]:
    """
    Identifie la nature des équilibres du système en fonction de la valeur de E.
    Version alternative utilisant la fonction jacobian de scipy.

    Args:
        E: Paramètre d'exploitation ou environnemental

    Returns:
        dict: Dictionnaire associant chaque équilibre à sa classification
              ('stable', 'instable' ou 'selle')
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

"""
for e_maj in Es:
    print(f"Pour E = {e_maj} :")
    classes = class_equilibrium(e_maj)
    for eq in classes.keys():
        print(f"  l'équilibre {eq} est de type {classes[eq]}")
"""

def ulim(u0:vector, f:Callable=equation, t_max=100, E=0.00):
    """
    Retourne la pseudo-limite d'une solution u de condition initiale u0.

    Args:
        u0: Condition initiale de la solution.
        f: Fonction définissant l'équation différentielle.
        t_max: Temps maximal pour la simulation (pseudo-infini).
        E: Paramètre d'exploitation ou environnemental.

    Returns:
        vector: La pseudo-limite de la solution.
    """
    # S'assurer que u0 est un vecteur 1-dimensionnel
    u0 = array(u0).flatten()

    try:
        # Créer une fonction qui encapsule f avec le paramètre E
        def f_with_E(t, v):
            return f(t, v, E)

        sol = solve_ivp(f_with_E, [0, t_max], u0, method='RK45', rtol=1e-8, atol=1e-8)
        return array(sol.y[:,-1])
    except Exception as e:
        print(f"Erreur lors de l'intégration: {e}")
        # En cas d'erreur, retourner la condition initiale
        return u0

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
    u0 = array(u0).flatten()
    equi1 = array(equi1).flatten()
    equi2 = array(equi2).flatten()

    u_final = ulim(u0, f, t_max, E)
    return norm(u_final - equi2) - norm(u_final - equi1)

def intercept_g(w:float, E:float, t_max:int=100)->float:
    """
    Calcule le g tel que (g,w) est l'intersection entre la droite y = w et la courbe séparatrice.

    Args:
        w (float): Valeur de la biomasse ligneuse
        E (float): Paramètre d'exploitation ou environnemental
        t_max (int, optional): Temps maximal pour la simulation. Defaults to 100.

    Returns:
        float: Valeur de g correspondant à l'intersection, ou None si aucune intersection n'existe.
    """
    # Récupération des classes d'équilibres
    classes = class_equilibrium(E)

    # Récupérer les équilibres stables
    stable = [eq for eq, cls in classes.items() if cls == 'stable']

    nb_stable = len(stable)

    # Un seul équilibre stable : pas de courbe séparatrice
    if nb_stable < 2:
        return None

    # On sait qu'il y a soit 1 ou 2 équilibres stables
    equi1, equi2 = stable

    def objective(g):
        # S'assurer que les vecteurs sont correctement formatés
        u0 = array([g, w]).flatten()
        equi1_array = array(equi1).flatten()
        equi2_array = array(equi2).flatten()
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
        tuple: (w, g) où w et g sont des tableaux NumPy contenant les coordonnées
              des points de la courbe séparatrice. Si aucune courbe séparatrice
              n'existe, retourne des tableaux vides.
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

def plot_separatrix(E:float):
    """
    Trace la courbe séparatrice (si elle existe) avec les équilibres vers lesquels
    toutes les trajectoires sont censées converger.

    Args:
        E: Paramètre d'exploitation ou environnemental
    """
    # Récupérer les points de la courbe séparatrice
    w, g = separatrix_points(E)

    # Récupérer les équilibres et leur classification
    eqs = equilibres(E)
    classes = class_equilibrium(E)

    # Créer la figure
    plt.figure(figsize=(10, 8))

    # Tracer la courbe séparatrice si elle existe
    if len(w) > 0:
        plt.plot( g,w, '--', color='blue', label='Courbe séparatrice')

    # Tracer les équilibres avec des couleurs différentes selon leur stabilité
    for eq in eqs:
        eq_tuple = tuple(eq)
        if eq_tuple in classes:
            if classes[eq_tuple] == 'stable':
                plt.plot(eq[0], eq[1], 'go', markersize=10, label='Équilibre stable')
            elif classes[eq_tuple] == 'instable':
                plt.plot(eq[0], eq[1], 'ro', markersize=10, label='Équilibre instable')
            else:  # selle
                plt.plot(eq[0], eq[1], 'bo', markersize=10, label='Point selle')

    # Ajouter une grille et des légendes
    plt.grid(True)
    plt.xlabel('herve grass g')
    plt.ylabel('herve weed  w')
    plt.title(f'Courbe séparatrice et équilibres pour E = {E}')

    # Éviter les doublons dans la légende
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Définir les limites du graphique
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Afficher le graphique
    plt.show()


