"""
Librairie d'analyse d'un système Rangeland dans le cas d'une équation  autonome et un espace d'état Ω = [0 , 1]²
Afin de répondre aux questions du projet , j'ai implémenté deux classe , une qui repond au question du projet et ses constantes , une autre pour un cadre général

Class:
    EquilibriumType : les differentes natures d'équilibre possibles
    Equilibrium : representations `utiles` d'un équilibre
    DefaultParams : les paramètres constants du système pra défaut


"""

from abc import ABC , abstractmethod
from enum import Enum
from dataclasses import dataclass , field
from typing import Callable, Tuple , NamedTuple , Any , Optional ,List
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq , fsolve
from scipy.differentiate import jacobian
from tools import phi


class EquilibriumType(Enum):
    """ Typle possibles d'équilibre """
    STABLE = "stable"
    UNSTABLE = "instable"
    SADDLE = "selle"

@dataclass(frozen=True)
class Equilibrium:
    """  répresentation d'un équilibre du système """
    coordinates : Tuple[float, float]  # coordonnées dans l'espace des états
    nature: Optional[EquilibriumType] = None  # nature de l'équilibre
    eigval: Optional[Tuple[complex , complex]] = None # valeuurs propres de la matrice jacobienne associée
    eigvect: Optional[Tuple[NDArray[np.float64] , NDArray[np.float64]]] = None # vecteurs propres asssociés au valeurs propres

class DefaultParams(NamedTuple):
    """ Paramètres constants  du système par défaut """
    Rg: float = 0.27
    Rw: float = 0.4
    a: float = 0.31
    b: float = 0.6
    c: float = 1.07

@dataclass
class RangelandSystem(ABC):
    """
    Classe abstraite définissant l'interface pour l'analyse d'un système Rangeland

    Attributs
    ---------
        params : l'ensemble des parmètres constants du sytème
        equation : fonction second membre du système (considèrant le système mis sous forme d'une équation de 1er ordre)

    """

    params:NamedTuple
    equation : Callable[[float , np.ndarray,Tuple[float,...]] , np.ndarray]

    @abstractmethod
    def find_equilibria(self,*args:Any)->List[Equilibrium]:
        """ Retourne la liste des  équilibres du système , mais avec juste leur coordonnées  """


    @abstractmethod
    def separatrix_points(self,*args)->Tuple[np.ndarray , np.ndarray]:
        """ clacule le point de la séparatrice"""

    def get_classified_equilibria(self, *args):
        """
        Liste des équilibre avec toutes leurs informations utiles
        """

        eqs = self.find_equilibria(*args)
        class_eq = []

        for eq in eqs:

            # matrice jacobienne au point d'équilibre
            jac = jacobian(
                lambda v : self.equation(0, v, args),
                np.array(eq.val)
                )

            eigenvals , eignvects = np.linalg.eig(jac)

            # Partie réelle des valeurs propres
            real_parts = np.real(eigenvals)

            # Les valeurs propres sont toutes de partie réelle négative
            if all(real_parts<0):
                nature = EquilibriumType.STABLE
            # Les valeurs propres sont toutes de partie réelle positive
            elif all(real_parts>0):
                nature= EquilibriumType.UNSTABLE
            else:
                nature = EquilibriumType.SADDLE
            class_eq.append(Equilibrium(
                coordinates= eq.coordinates ,
                nature=nature,
                eigval=tuple(eigenvals),
                eigvect=tuple(eignvects.T)
                ))
        return class_eq




@dataclass
class DefaultSystem(RangelandSystem):
    """ implémentation du sytème Rangeland étudié dans le projet"""

    params:DefaultParams = field(default_factory=DefaultParams())

    equation:Callable[[float , np.ndarray,Tuple[float,...]] , np.ndarray] = field(init=False)

    def __post_init__(self):
        self.equation = self._default_equation

    def _default_equation(self, t: float,
                          state: np.ndarray,
                          E:float=0.0) -> np.ndarray:
        """
            Membre de droite du système différentiel
        Args:
            t: Temps (non utilisé mais requis par solve_ivp)
            state: Vecteur d'état [g, w]
            E: coefficient de proportionalité à la taille du bétail
        """
        g, w = state
        p = self.params
        if E == 0 and g == 0:
            dgdt = p.Rg*g*(1-g- 0.6*w)
            dwdt = p.Rw*w*(1-w-1.07*g)
        else:
            dgdt = p.Rg*g*(1-g- p.b*w*((E + g)/(p.a*E + g)))
            dwdt = p.Rw*w*(1-w- p.c*g*((p.a*E + g)/(E + g)))

        return np.array([dgdt , dwdt])

    def _dubble_u(self, g:float , E:float)->float:
        """ w en fonction de g dans le cass par défaut"""

        p = self.params

        return 1- p.c*g*((p.a*E + g)/(E+g))

    def find_equilibria(self, E:float)->List[Equilibrium]:

        p = self.params
        E1 , E2 = p.E1 , p.E2

        # les équilibres triviaux
        equilibrium = [
            Equilibrium((0,0)),
            Equilibrium((1, 0)),
            Equilibrium((0, 1))
            ]

        # coefficients du polynômes en g
        a1 , b1 , c1 = (1 - p.c*p.b) , (1-p.c*p.b)*p.a*E - (p.b -1 ) , 0.29*E
        delta = b1**2 -4*a1*c1
        if E1 < E < E2: # pas d'équilibres non triviaux
            return equilibrium

        elif E==E1 or E==E2: # un seul équilibre non trivial
            g = -b1/(2 * a1)
            w = self._dubble_u(g , E)
            #if 0 < g < 1 and 0 < w1 < 1:
            equilibrium.append((g, w))
        else  :     # deux équilibres non triviaux
            g1 = (-b1 - np.sqrt(delta)) / (2 * a1)
            g2 = -b1/a1 - g1
            w1 , w2 = self._dubble_u(g1,E) , self._dubble_u(g2,E)

            # on ne garde que ceux dans [0,1]²
            if 0 < g1 < 1 and 0 < w1 < 1:
                equilibrium.append((g1, w1))
            if 0 < g2 < 1 and 0 < w2 < 1:
                equilibrium.append((g2, w2))

        return equilibrium

    def _intercept_g(self , w:float, E:float, t_max:int=100)->float:
        """
        Calcule le g tel que (g,w) est l'intersection entre la droite y = w et la courbe séparatrice.

        Args:
            w : pourcentage de mauvaises herbes
            E : proportion de bétail
            t_max : borne maximale de temps

        Returns:
            float: Valeur de g correspondant à l'intersection, ou None si aucune intersection n'existe.
        """
        # Récupération des classes d'équilibres
        classified = self.get_classified_equilibria(E)

        # Récupérer les équilibres stables
        stable = [eq.val for eq  in classified if eq.nature == 'stable']

        nb_stable = len(stable)

        # Un seul équilibre stable : pas de courbe séparatrice
        if nb_stable < 2:
            return None

        # On sait qu'il y a soit 1 ou 2 équilibres stables
        equi1, equi2 = stable

        def objective(g):
            # S'assurer que les vecteurs sont correctement formatés
            u0 = np.array([g, w])
            equi1_array = np.array(equi1)
            equi2_array = np.array(equi2)
            return phi(u0, equi1_array, equi2_array, self.equation, t_max, (E , ))

        try:
            return brentq(objective, 0, 1)
        except ValueError:
            # Si brentq ne trouve pas de solution, retourner None
            return None

    def separatrix_points1(self, E:float , grid_size:int=200):

        """
        Génère des points de la séparatrice
        Args:
            E: Paramètre d'exploitation ou environnemental
            grid_size: Nombre de points à générer. Defaults to 200.

        Returns:
            tuple: (w, g) où w et g sont des tableaux NumPy contenant les coordonnées des points de la courbe séparatrice. Si aucune courbe séparatrice n'existe, retourne des tableaux vides.
        """
        # On se fixe un ensemble de w entre 0 et 1
        w_grid = np.linspace(0,1,grid_size)

        # On trouve le g* de l'intersection entre la courbe et y = w
        g_separatrix = [self._intercept_g(w,E,t_max=500) for w in w_grid]

        # Filtrer les valeurs None
        valid_indices = [i for i, g in enumerate(g_separatrix) if g is not None]

        if not valid_indices:
            # Aucune courbe séparatrice n'existe
            return np.array([]), np.array([])

        # Extraire les valeurs valides
        w_valid = w_grid[valid_indices]
        g_valid = np.array([g_separatrix[i] for i in valid_indices])

        return w_valid, g_valid

    def find_extinction_of_g(self,*args):
        """ permet de trouver s'il existent , les paramètres ``args`` en dessus ou en dessous desquels toutes les trajectoire mènent à l'extinction de ``g``

        concretement cela revient à determiner la plage des args pour lesquels (0,w*) (pour un w* qui forme avec 0 un équilibre du système) est le seul equilibre stable du système

        """
    @abstractmethod
    def find_extinction_of_w(self,*args):
        """ permet de trouver s'il existent , les paramètres ``args`` en dessus ou en dessous desquels toutes les trajectoire mènent à l'extinction de ``w``

        concretement cela revient à determiner la plage des args pour lesquels (g*,0) (pour un g* qui forme avec 0 un équilibre du système) est le seul equilibre stable du système

        """

@dataclass
class GeneralSystem(RangelandSystem):
    """ implémentation d'un autre système autonome contenunsur [0,1]² """

    def find_equilibria(self,*args:Tuple[float,...]):

       # Points initiaux pour la recherche des équilibres
        initial_guesses = [
            np.array([0.0, 0.0]),  # origine
            np.array([1.0, 0.0]),  # axes
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),  # point intérieur
            np.array([0.2, 0.8]),  # autres points potentiels
            np.array([0.8, 0.2])
        ]

        equilibria = []
        for guess in initial_guesses:
            # Résolution numérique
            sol = fsolve(lambda u : self.equation(0,u,), guess, full_output=True)
            if sol[2] == 1:  # La convergence a réussi
                eq_point = sol[0]
                # Vérification des conditions physiques
                if all(0 <= x <= 1 for x in eq_point):
                    # Vérification que ce point n'est pas déjà trouvé
                    if not any(np.allclose(eq_point, eq.coordinates) for eq in equilibria):
                        equilibria.append(Equilibrium(
                            tuple(eq_point)))

