"""
Ce programme de traiter/étudier l'équation différentielle d'un solide suspendu à la verticale à un ressort

    (1) m.x" = -k.x -ε.x³ - μ|x'|x' + m.g

    où :
    - x : elongation du ressort / position du solide sur l'axe vertical
    - μ : coefficient de proportionalité des frotements par rapport à la vitesse ?
    - m : masse du solide
    - g : constante de gravitation terrestre
    - k , ε : constantes de la reponse f(x) = -k.x -ε.x³ du ressort
"""

from typing import  Union , Dict , Optional
from scipy.integrate import solve_ivp
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


# DEFINITION DES CONSTANTES

m , k , g = 1 , 3 , 9.81

# DEFINITION DU SYSTEME
def system(t:float ,u:NDArray,epsilon:float,mu:float=0)->NDArray:
    """
        membre de droite de l'équation différentielle ramenée à un système d'ordre 1
    """
    x , dx = u
    return np.array([ dx ,
                     (- k*x - epsilon*x**3 - mu*abs(dx)*dx )/m + g
                    ])


def periode(epsilon:float ,mu:float = 0, t_max:float = 50, y0 : Optional[NDArray] = None , info:Optional[bool] = False)->Union[float , Dict]:
    """
    ## Calcule la période de la soltuion u dans le cas sans frottement , en detectant les maxima de x (dx = 0 avec direction négative)

    Strategie
    ---------
        Comme x(t) et ∂ₜx(t)  périodiques de  période commune T , on repère les temps tᵢ  entre deux extrema de x (qui correspondent aux racines de ∂ₜx(t) ) , puis on , prend les différences entre Tᵢ = tᵢ₊₁ - tᵢ qui sont supposées tendre en moyenne vers T , et afin on prend la moyenne des Dᵢ . Pour cela :

        -  ``event()`` : permet de repérer les temps tᵢ
        - ``event.direction`` : spécifie dans quel sens le chagement de signe se fait
    """
    if y0 is None:
        y0 = np.array([0.0,0.0]) # condition initiale par défaut

    # cas où ε = 0
    if epsilon == 0 and mu ==0 :
        period = 2*np.pi*np.sqrt(m/k)
        if not info :
            return period
        sol = solve_ivp(system , (0,period*3) , y0 , args =(epsilon , mu) , dense_output= True )
        return  {'period':period , 'sol':sol}

    # Cas général
    ## Evènement : détection des maxima (dx = 0 descendant)
    def event(t:float, y:NDArray,epsilon:float,mu:float)->float:
        return y[1]

    event.terminal = False
    event.direction = -1 # passage positif au négatif : c'est minimum

    sol = solve_ivp(system, (0,t_max) , y0 , args=(epsilon , mu), events=event , dense_output=True ,method='RK45',rtol=1e-8 , atol=1e-10)

    # Identifier les temps des événements
    t_events = sol.t_events[0]

    if len(t_events)<2:
        period = None # pas assez de maxima détectés
    else :
        #Calculer les périodes
        periods = np.diff(t_events)
        # Période moyenne
        period = np.mean(periods) if len(periods)>0 else None

    if info :
        return {'period':period , 'sol':sol}
    return period


def vitesse_critique(epsilon:float, mu:float=0.0 , x_max:float = 5.0 , tol:float=1e-6 , v0_max:float=50)->float:
    """
    Determine la vitesse initiale maximale par recherche binaire

    Strategie
    ---------
    - pour **mu** = 0 : utilisation de la conservation de l'énergie
    - pour **mu** > 0 : recherche binaire avec simulation numérique

    Parameters
    ----------

    x_max : float, optional
        position maximal à ne pas dépaser
    tol : float, optional
        pécision
    v0_max : float, optional
        _description_, by default 50

    Returns
    -------
    float
        _description_
    """
    if mu == 0:
        #Solution analytique via l'énergie mécanique
        def energy(x):
            return 0.5*k*x**2 +0.25*epsilon*x**4 - m*g*x

        E_max = energy(x_max)
        v0 = np.sqrt(2*(E_max - energy(0))/m)
        return v0
    else :
        # recherche dichotomique
        v_low , v_high = 0.0 , v0_max
        best_v = 0.0

        for _ in range(20):
            v_mid = (v_low + v_high)/2
            sol = solve_ivp(system , (0,100),[0.0,v_mid] , args=(epsilon , mu) , rtol = 1e-7 , atol = 1e-9)

            x_max_sim = np.max(sol.y[0])
            if x_max_sim < x_max + tol:
                best_v = v_mid
                v_low = v_mid
            else:
                v_high = v_mid

        return best_v

#Visualisation de l'évolution des positions x et vitesses dx pour différentes valeurs de epsilon
"""
NUM_EPSILON_VALUES = 9
TIME_MAX = 10
NUM_TIME_POINTS = 100

epsilon_values = np.linspace(0,1,NUM_EPSILON_VALUES)
time_points = np.linspace(0,TIME_MAX,NUM_TIME_POINTS)
NUM_ROWS = NUM_EPSILON_VALUES//3

# Création de la grille de graphiques
fig , ax = plt.subplots( NUM_ROWS , 3 , figsize = ( 18 , 10))

# Tracé des courbes
for i , ep  in enumerate(epsilon_values) :
    row , col = i//3 , i%3
    solt = periode(ep , info=True)
    solution = solt["sol"].sol(time_points)
    T = solt["period"] # période calculée
    v0 = vitesse_critique(ep,mu=0.1)
    if np.isfinite(T) and T > 0 :

        # Création des points périodiques ()
        n_periods = 3
        t_periodic = np.arange(0,n_periods*T , T)

        # Tracé position et vitesse
        ax[row,col].plot( time_points , solution[0]  , label = 'position x(t)' )
        ax[row,col].plot( time_points , solution[1] , label = " vitesse $\partial_t x(t)$")
        ax[row,col].scatter(t_periodic , np.zeros_like(t_periodic),marker = 'o' , label = 'points périodiques')
        ax[row , col].scatter(0,0, color = 'black' , label=f"vitesse critique pour ε = {ep:.2f} et μ = 0.1  = {v0:.2f}")
        ax[row,col].set_title(f"ε = {ep:.1f}")
        ax[row,col].grid(True)
        ax[row,col].legend()
    else :
        print(f"Période invalide pour epsilon = {ep:.1f} : T = {T} ")
plt.tight_layout()
plt.show()
"""

def plot_period(num_points:int):

    """
    graph de la période en fonction de ε
    """
    epsilon_values = np.linspace(0,1,num_points)
    Periods = [periode(ep) for ep in epsilon_values]

    plt.plot(epsilon_values , Periods , label = 'periode en fonction de ε ')
    plt.ylabel('period')
    plt.xlabel('ε ')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_solution(epsilon : float , mu:float=0.0 , t_sim : int = 100 , num_points:int = 500):
    """
    graphe de la x et dx

    Args:
        epsilon (float): [description]
        mu (float, optional): [description]. Defaults to 0.0.
    """
    t = np.linspace(0,t_sim,num_points)
    solt = periode(epsilon=epsilon , mu=mu , t_max= t_sim , info=True)
    solution = solt['sol'].sol(t)
    x = solution[0]
    dx = solution[1]

    plt.plot(t , x , label = 'position x(t)')
    plt.plot(t , dx , label='vitesse dx(t)/dt')
    plt.legend()
    plt.grid(True)
    plt.show()

def x_max_sim(epsilon:float , mu : float , v0:float , t_sim :float= 100):
    """
    AI is creating summary for x_max_sim

    Args:
        epsilon (float): [description]
        mu (float): [description]
        v0 (float): [description]
        t_sim (float, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    sol = periode(epsilon, mu = mu , y0 = [0,v0],t_max= t_sim  , info=True)['sol']

    # retourner l'élongation maximale
    return np.max(sol.y[0])


print(x_max_sim(0.3,mu=0.1 , v0=0.9))