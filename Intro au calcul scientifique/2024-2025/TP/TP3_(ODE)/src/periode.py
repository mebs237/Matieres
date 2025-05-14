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



class ResultPeriod:
    """
    classe pour encapsuler les résultats de la période
    """
    def __init__(self, period, sol, y0, found:bool=True, message=""):
        self.period = period #
        self.sol = sol
        self.y0 = y0
        if period is None:
            found = False
        self.found = found
        self.message = message

    def __repr__(self):
        return (f"ResultatPeriode(period={self.period}, status='{self.found}', "
                f"message='{self.message}')")


def periode(epsilon:float ,mu:float = 0,
            t_max:float = 50,
            y0 : Optional[NDArray] = None
            )->ResultPeriod:
    """
    Calcule la période/pseudo-période de la soltuion u dans le cas sans frottement (μ = 0) et avec frottement (μ > 0)

    Parameters
    ----------
    epsilon : float
        paramètre de la force de rappel du ressort
    mu : float, optional
        paramètre de frottement
    t_max : float, optional
        temps maximum de simulation
    y0 : NDArray, optional
        condition initiale

    """
    if y0 is None:
        y0 = np.array([0.0,0.0]) # condition initiale par défaut

    # Evènement : détection des maxima (∂ₜx = 0 )
    def event(t:float, y:NDArray,epsilon:float,mu:float)->float:
        return y[1]

    event.terminal = False
    event.direction = -1 # passage positif au négatif : c'est maximum

    sol = solve_ivp(system, (0,t_max) , y0 , args=(epsilon , mu), events=event , dense_output=True ,method='RK45',rtol=1e-10 , atol=1e-12)

    # Identifier les temps des événements
    t_events = sol.t_events[0]

    if len(t_events)<2:
        return ResultPeriod(None,sol,
                            y0,
                            message="pas assez d'extrema détectés ,augmenter t_max ou epsilon")

    else :
        #Calculer les périodes
        periods = np.diff(t_events)
        # Période moyenne
        period = np.mean(periods)
        return ResultPeriod(period ,sol ,y0)

def energy(x , epsilon):
    """
    energie potentielle en la position x
    """
    return 0.5*k*x**2 +0.25*epsilon*x**4 - m*g*x

class ResultVitesse:
    """
    classe pour encapsuler les résultats de la vitesse critique
    """
    def __init__(self, v_crit, message=""):
        self.v_crit = v_crit
        self.message = message

    def __repr__(self):
        return (f"ResultatVitesse(v_crit={self.v_crit}, add_info={self.message})")

def vitesse_critique(epsilon:float, mu:float=0.0 ,
                     x_target:float = 5.0 , atol:float=1e-12 ,
                     v0_max:float=50 , n_iter : int = 100
                     )->ResultVitesse:
    """
    Determine la vitesse initiale maximale pour que la position x ne dépasse pas ``x_target``


    Parameters
    ----------

    x_target : float, optional
        position maximal à ne pas dépaser
    atol : float, optional
        pécision souhaitée pour la vitesse
    v0_max : float, o
        borne maximale pour la recherche dichotomique
    n_iter : int, optional
        nombre d'itérations maximum pour la recherhce dichotomique

    """
    if mu == 0:
        #Solution analytique via l'énergie mécanique
        E_max = energy(x_target,epsilon=epsilon)
        # v0_max = vitesse initiale maximale
        v0 = np.sqrt(2*E_max /m)
        return ResultVitesse(v_crit=v0 , message="pas de frottement μ = 0")
    else :
        # recherche dichotomique
        v_low , v_high = 0.0 , v0_max
        v_mid = (v_low + v_high)/2

        i = 0

        condv = abs(v_high-v_low)<=atol

        while (not condv) and i<n_iter:

            v_mid = (v_low + v_high)/2
            sol = solve_ivp(system , (0,100),[0.0,v_mid] , args=(epsilon , mu) , rtol = 1e-10 , atol = 1e-12)
            # position maximal partant initialement de v_mid
            x_maxs = np.max(sol.y[0])

            condx = abs(x_maxs - x_target)<=atol
            if condx :
                return ResultVitesse(v_crit=v_mid , message=f"vitesse  trouvée après  {i}/{n_iter} ")

            elif x_maxs < x_target:
                v_low = v_mid
            else:
                v_high = v_mid
            i+=1
            condv = abs(v_high-v_low)<=atol

        return ResultVitesse(v_crit=v_mid ,
                             message=f"vitesse trouvée après {i}/{n_iter} itérations")

def plot_period(num_points:int):

    """
    graph de la période en fonction de ε
    """
    epsilon_values = np.linspace(0,1,num_points)
    Periods = [periode(ep).period for ep in epsilon_values]

    plt.plot(epsilon_values , Periods , label = 'periode en fonction de ε ')
    plt.ylabel('period')
    plt.xlabel('ε ')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_solution(epsilon : float , mu:float=0.0 ,y0 = None, t_sim : int = 60 , num_points:int = 200):
    """
    graphe de la x et dx

    Args:
        epsilon (float): [description]
        mu (float, optional): [description]. Defaults to 0.0.
    """
    t = np.linspace(0,t_sim,num_points)
    if y0 is not None:
        solt = periode(epsilon=epsilon , mu=mu , t_max= t_sim , y0=y0 )
    else :
        solt = periode(epsilon=epsilon , mu=mu , t_max= t_sim )
        y0 = solt.y0
    solution = solt.sol.sol(t)
    x = solution[0]
    dx = solution[1]
    x_max = x_max_sim(epsilon=epsilon , mu=mu , v0=y0[1] , t_sim=t_sim)
    plt.figure(figsize=(10,5))

    plt.axhline(x_max,color = 'red' , lw = 0.5)
    plt.plot(t , x , label = 'position x(t)')
    plt.plot(t , dx , label='vitesse dx(t)/dt')
    plt.title(f"ε = {epsilon:.2f} , μ = {mu:.2f}")
    plt.xlabel('t')
    plt.legend()
    plt.grid(True)

    plt.show()

def x_max_sim(epsilon:float , mu : float , v0:float , t_sim :float= 100):
    """
    position maximale atteinte partant de la vitesse initiale v0
    """
    sol = periode(epsilon, mu = mu , y0 = [0,v0],t_max= t_sim ).sol

    # retourner l'élongation maximale
    return np.max(sol.y[0])




