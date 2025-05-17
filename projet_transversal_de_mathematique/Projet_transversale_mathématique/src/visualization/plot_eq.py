from src.models.Equilibres import equilibres, edo  # Replace with the actual names you need to import
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from typing import *
from numpy import cos , sin
import scipy.integrate as sp



Vector = Tuple[float, float]

e_majs = [0,
          0.1,
          0.3,
          0.4]

equilibriums = [(0,0),(1,0),(0,1)]

# On remet tous les équilibres pour les valeurs de E (e_majs) considérées
# dans une liste pour pouvoir les tracer à l'écran

for i in range(len(e_majs)):
    eq_inter = equilibres(e_majs[i])
    for eq in eq_inter:
        if eq not in equilibriums:
            equilibriums.append(eq)

def in_0_1(v: Vector) -> bool:
    """
    Vérifie si le vecteur v est dans le carré [0,1]**2
    """
    return (v[0] >= 0 and v[0] <= 1) and (v[1] >= 0 and v[1] <= 1)



def rangeland(g0 : float,w0 : float, e_maj:float) -> sp.OdeSolution:
    """
    Résolution de l'EDO du rangeland de conditions initiales g(0) = g0 et w(0) = w0
    """

    v0 = (g0,w0)
    t_int = [0,100000]
    sol = sp.solve_ivp(lambda t,v : edo(t,v,e_maj), t_int, v0)
    return sol

def plot_rangeland(sol: Optional[sp.OdeSolution], e_maj:float) -> None:
    """
    Permet de tracer les graphiques des solutions de l'EDO
    """

    n_points = len(sol.t)
    g = sol.y[0]
    w = sol.y[1]

    curve, = plt.plot(g,w) # Tracé de la solution
    color = curve.get_color()
    # Ajout des flèches

    mid_index = len(sol.t)//20
    plt.annotate('', xy=(g[mid_index], w[mid_index]),
                 xytext=(g[mid_index - 1], w[mid_index - 1]),
                arrowprops=dict(arrowstyle='->', color=color))

def near_equilibrium(eq: Vector, n:int, ε : float = 0.1) -> List[Vector]:
    """
    Trouve n points (g,w) ε-proches du point eq en
    découpant le cercle de rayon ε centré en (g,w) en n points
    """
    return [(eq[0]+ ε*cos(2*sin*i/n),eq[1]+ ε*sin(2*sin*i/n))
            for i in range(n)]


# Tracé des solutions

decoup1 = [0,0.25,0.5,0.75,1]
decoup2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1]
cond_init_extinction = [(g,w) for g in decoup1 for w in [0,1]]
cond_init_extinction2 = [(g,w) for g in decoup2 for w in decoup2]
cond_init_eq = [near_equilibrium(eq, 30,10e-10)[i] for eq in [equilibriums[2]]
                for i in range(len(near_equilibrium(eq,30,10e-10)))]

"""for e_maj in e_majs:
    for ci in cond_init_extinction:
        if in_0_1(ci):
            sol = rangeland(ci[0],ci[1],e_maj)
            plot_rangeland(sol,e_maj)

        # Tracé du carré [0,1]**2

    axis = plt.gca()
    square = plt.Rectangle((0,0), 1, 1, fill=None, edgecolor='gray', linewidth=2)
    axis.add_patch(square)

    # Tracé des équilibres

    plt.scatter([eq[0] for eq in equilibriums],
    [eq[1] for eq in equilibriums],
    color = 'blue',
    label = 'Equilibres')
    plt.scatter([ci[0] for ci in cond_init_extinction if in_0_1(ci)],
    [ci[1] for ci in cond_init_extinction if in_0_1(ci)],
    color = 'black',
    label = 'Conditions initiales')
    plt.title(f'Rangeland pour E = {e_maj}')
    plt.xlabel('g')
    plt.ylabel('w')
    plt.grid()
    plt.legend()
    plt.show()"""

"""for e_maj in e_majs:
    for ci in cond_init_extinction2:
        if in_0_1(ci):
            sol = rangeland(ci[0],ci[1],e_maj)
            plot_rangeland(sol,e_maj)

        # Tracé du carré [0,1]**2

    axis = plt.gca()
    square = plt.Rectangle((0,0), 1, 1, fill=None, edgecolor='gray', linewidth=2)
    axis.add_patch(square)

    # Tracé des équilibres

    plt.scatter([eq[0] for eq in equilibriums],
    [eq[1] for eq in equilibriums],
    color = 'blue',
    label = 'Equilibres')
    plt.scatter([ci[0] for ci in cond_init_extinction2 if in_0_1(ci)],
    [ci[1] for ci in cond_init_extinction2 if in_0_1(ci)],
    color = 'black',
    label = 'Conditions initiales')
    plt.title(f'Rangeland pour E = {e_maj}')
    plt.xlabel('g')
    plt.ylabel('w')
    plt.grid()
    plt.legend()
    plt.show()"""

