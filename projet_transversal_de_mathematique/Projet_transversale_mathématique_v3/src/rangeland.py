"""
Librairie d'analyse d'un système Rangeland dans le cas d'une équation  autonome et un espace d'état Ω = [0 , 1]²

J'ai implémenté ``DefaultSystem`` pour répondre aux questions du projet et ses constantes , et ``GeneralSystem`` pour essauyé de généraliser
"""

from abc import ABC , abstractmethod
from enum import Enum
from dataclasses import dataclass , field
from typing import Callable, Tuple , NamedTuple , Any , Optional ,List
import numpy as np
from collections import namedtuple
from numpy.typing import NDArray
from jax.numpy import array as jarray
from jax import jacobian
from scipy.optimize import brentq , fsolve
from scipy.differentiate import jacobian
from tools import phi , solve_system


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
    eigvect: Optional[Tuple[
        NDArray[np.float64] ,
        NDArray[np.float64]
        ]] = None # vecteurs propres asssociés au valeurs propres

    @property
    def g(self):
        """ retourne la 1ère coordonnées g"""
        return self.coordinates[0]

    @property
    def w(self):
        """ retourne la 2e coordonnée w"""
        return self.coordinates[1]


class DefaultParams(NamedTuple):
    """ Paramètres constants  du système par défaut """
    Rg: float = 0.27
    Rw: float = 0.4
    a: float = 0.31
    b: float = 0.6
    c: float = 1.07

# object encapsulant les informations d'une portion de la séparatrice
Portion = namedtuple('Portion',[
    't_arret',  # temps d'arrêt
    'direction',    # stable_plus , ou stable_minus
    'epsilon',      # amplitude de la pertubation
    'event_type',   # 'point selle' , 'bord'
    'u_traj',       # tableau des points (2 x N)
    't_traj'        # tableau des temps
])

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
    def find_equilibria(self,*args)->List[Equilibrium]:
        """ Retourne la liste des  équilibres du système , mais avec juste leur coordonnées  """

    #@abstractmethod
   # def separatrix_points(self,*args)->Tuple[np.ndarray , np.ndarray]:
        """ clacule le point de la séparatrice"""

    def get_classified_equilibria(self, *args):
        """
        Liste des équilibre avec toutes leurs informations utiles
        """

        eqs = self.find_equilibria(*args)
        class_eq = []

        for eq in eqs:

            # matrice jacobienne au point d'équilibre
            jac = self.compute_jacobian(self,
                                        np.array(eq.coordinates),args)

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

    def compute_jacobian(self, u:NDArray,*args)->NDArray:
        """" calcule la jacobienne de ``self.equation`` en u"""
        u_jax = jarray(u)
        def func(u_j):
            u_np = np.array(u_j)
            return jarray(self.equation(0,u_np,*args))
        jac = jacobian(func , u_jax)
        return np.array(jac)

@dataclass
class DefaultSystem(RangelandSystem):
    """ implémentation du sytème Rangeland étudié dans le projet"""

    params:DefaultParams = field(default=DefaultParams())

    equation:Callable[[float , np.ndarray,Tuple[float,...]] , np.ndarray] = field(init=False)

    def __post_init__(self):
        def default_equation(t: float,
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
                dgdt = 0
                dwdt = p.Rw*w*(1-w)
            else:
                dgdt = p.Rg*g*(1-g- p.b*w*((E + g)/(p.a*E + g)))
                dwdt = p.Rw*w*(1-w- p.c*g*((p.a*E + g)/(E + g)))

            return np.array([dgdt , dwdt])

        self.equation = default_equation


    def _dubble_u(self, g:float , E:float)->float:
        """ w en fonction de g dans le cass par défaut"""

        p = self.params

        return 1- p.c*g*((p.a*E + g)/(E+g))

    def delta(self,E:float):
        """Discriminant du polynôme en g"""
        p =self.params
        return ( p.b -1 +  (1-p.c*p.b)*p.a*E)**2 - 4*(p.b - p.a)((1 - p.c*p.b))
    def find_equilibria(self, E:float)->List[Equilibrium]:
        """calcule les coordonnées de équilibres du systeme"""
        p = self.params

        # les équilibres triviaux
        equilibrium = [
            Equilibrium((0,0)),
            Equilibrium((1, 0)),
            Equilibrium((0, 1))
            ]

        # coefficients du polynômes en g
        a1 , b1 = (1 - p.c*p.b) , (1-p.c*p.b)*p.a*E + (p.b - 1 )

        if self.delta(E)<0: # pas d'équilibres non triviaux
            return equilibrium

        elif self.delta(E)==0: # un seul équilibre non trivial
            g = -b1/(2 * a1)
            w = self._dubble_u(g , E)
            #if 0 < g < 1 and 0 < w1 < 1:
            equilibrium.append(Equilibrium((g,w)))

        else  :     # deux équilibres non triviaux
            g1 = (-b1 - np.sqrt(self.delta(E))) / (2 * a1)
            g2 = -b1/a1 - g1
            w1 , w2 = self._dubble_u(g1,E) , self._dubble_u(g2,E)

            # on ne garde que ceux dans [0,1]²
            if 0 < g1 < 1 and 0 < w1 < 1:
                equilibrium.append(Equilibrium((g1,w1)))
            if 0 < g2 < 1 and 0 < w2 < 1:
                equilibrium.append(Equilibrium((g2, w2)))

        return equilibrium

    def compute_jacobian(self,u:NDArray,E:float)->NDArray:
        """ redéfinie pour plus de précision """
        g,w=u
        p = self.params
        if E==0 and g==0:
            d1dg , d1dw = .0 , .0
            d2dg, d2dw = .0 , p.Rw(1-2*w)
        else :
            d1dg = p.Rg*( 1 - 2*g - ((p.b*w)/(p.a*E+g))*(E+g - (p.a-1)*E/(p.a+g)))
            d1dw = -p.Rg*g*p.b((E+g)/(p.a*E + g ))
            d2dg = ((-p.Rw*p.c*w)/(E+g))*( p.a*E +g (g*(1-p.a)*E)/(E+g))
            d2dw = p.Rw*(1-2*w - p.c*g*((p.a*E+g)/(E+g)))

        return np.array([[ d1dg , d1dw],
                        [ d2dg , d2dw]])

    def _intercept_g(self , w:float, E:float, t_max:int=100)->float:
        """
        Calcule le g* tel que (g*,w) soit sur la courbe séparatrice.

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
        stable = [eq.val for eq  in classified if eq.nature == EquilibriumType.STABLE]

        nb_stable = len(stable)

        # Un seul équilibre stable : pas de courbe séparatrice
        if nb_stable < 2:
            return None

        # On sait qu'il y a soit 1 ou 2 équilibres stables
        equi1 , equi2 = stable[0] , stable[1]

        def objective(g):
            # S'assurer que les vecteurs sont correctement formatés
            u0 = np.array([g, w])
            coord1 = np.array(equi1.coordinates)
            coord2 = np.array(equi2.coordinates)
            return phi(u0, coord1, coord2 , self.equation, t_max, (E , ))

        try:
            return brentq(objective, 0, 1)
        except ValueError:
            # Si brentq ne trouve pas de solution, retourner None
            return None

    def separatrix_points_01(self, E:float , grid_size:int=200):

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

    def separatrix_points(self , E:float , grid_size:int=200 ):
        return self.separatrix_points_01(E , grid_size=grid_size)
    def find_extinction_of_g(self,*args):
        """ permet de trouver s'il existent , les paramètres ``args`` en dessus ou en dessous desquels toutes les trajectoire mènent à l'extinction de ``g``

        concretement cela revient à determiner la plage des args pour lesquels (0,w*) (pour un w* qui forme avec 0 un équilibre du système) est le seul equilibre stable du système

        """
        return .0
    def find_extinction_of_w(self,*args):
        """ permet de trouver s'il existent , les paramètres ``args`` en dessus ou en dessous desquels toutes les trajectoire mènent à l'extinction de ``w``

        concretement cela revient à determiner la plage des args pour lesquels (g*,0) (pour un g* qui forme avec 0 un équilibre du système) est le seul equilibre stable du système

        """
        return .0

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
            np.array([0.8, 0.2]),
            np.array([0.3, 0.7]),
            np.array([0.7, 0.3])
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
        return equilibria

    def _find_separatrix_portion(self,eq:Equilibrium,
                                 others:Equilibrium,
                                 *args,
                                 epsilon:float =1e-3,
                                 t_max = 100,
                                 n_points = 1000)->Portion:
        """ trouve les portions de la séparatrice en intégrant à rebours depuis des pertubations dans la direction stable d'un point de selle """

        u_s = np.array(eq.coordinates)
        j_eigval , j_eigvect = eq.eigval , eq.eigvect

        #direction stable = valeur à partie réelle négative
        i_stable = np.argmin([np.real(ev) for ev in j_eigval])
        vs = np.real(j_eigvect[:,i_stable])
        vs /= np.linalg.norm(vs)  # normaliser le vecteur

        def event_bord(t,u,agrs):
            g,w = u
            if 0<=g<=1 and 0<=w<=1:
                return 1.0
            return 0.0
        even_bord