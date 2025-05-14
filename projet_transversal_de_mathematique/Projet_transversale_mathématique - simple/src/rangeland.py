"""
Librairie d'analyse d'un système Rangeland dans le cas d'une équation  autonome et un espace d'état Ω = [0 , 1]²
Afin de répondre aux question du projet , j'ai implémenté en certaines parties , un cas par defaut (celui de ce projet) et un cas général


"""


from typing import Callable, Tuple , NamedTuple , Dict , Any , Optional ,List , Literal
from dataclasses import dataclass , field
import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import brentq
from scipy.differentiate import jacobian
from scipy.optimize import fsolve
from .numericals_tools import phi , solve_system

@dataclass(frozen=True)
class ModelParameters:
    """classe définissant les paramètres constants du système différentiel"""
    # Paramètres de base
    Rg: float = 0.27 # taux d'accroissement de g
    Rw: float = 0.4 # taux d'accroissement de w
    a: float = 0.31
    b: float = 0.6  #
    c: float = 1.07
    # Paramètres additionnels
    _extra_params: Dict[str, Any] = field(default_factory=dict)

    def __getattribute__(self, name: str) -> Any:
        """Accède aux attributs, qu'ils soient dans les paramètres de base ou additionnels"""
        try:
            # Essaie d'abord d'accéder aux attributs de base
            return super().__getattribute__(name)
        except AttributeError:
            # Si l'attribut n'existe pas, cherche dans _extra_params
            extra_params = super().__getattribute__('_extra_params')
            if name in extra_params:
                return extra_params[name]
            print(f"Paramètre '{name}' non trouvé")

# les types d'équilibres
Equilibriumtype = Literal['stable','instable','selle']

# définition des équilibres d'un sytème
Equilibrium = NamedTuple('Equilibrium',[
    ('val' , Tuple[float,float]),   # valeurs
    ('nature' , Optional[Equilibriumtype]),       # type de l'équilibre
    ('eigval' , Optional[Tuple[complex , complex]]) # directions
])

@dataclass
class RangelandSystem:
    """Système complet de dynamique des pâturages"""

    _params:ModelParameters = field(default_factory=ModelParameters)
    @property
    def Rg(self) -> float:
        return self._params.Rg
    @property
    def Rw(self) -> float:
        return self._params.Rw
    @property
    def a(self) -> float:
        return self._params.a
    @property
    def b(self) -> float:
        return self._params.b
    @property
    def c(self) -> float:
        return self._params.c
    def get_param(self, key: str, default: Any = None) -> Any:
        """Accès aux paramètres additionnels"""
        return self._params.get(key, default)

    # Valeurs critiques de E (calculées une fois)
    E1: float = 0.3199208673727
    E2: float = 40.6057908272657

    # Fonction du sytème différentiel
    equation : Callable[[float , np.ndarray , float],np.ndarray]=field(default=None)

    def __post_init__(self):
        if self.equation is None:
            self.equation = self._default_equation

    def _default_equation(self, t: float, state: np.ndarray, args=(0.0 , )) -> np.ndarray:
        """
            Membre de droite du système différentiel
        Args:
            t: Temps (non utilisé mais requis par solve_ivp)
            state: Vecteur d'état [g, w]
            args: Paramètres supplémentaires (E , etc.)
        """
        g, w = state
        E = args[0]
        p = self._params
        if E == 0 and g == 0:
            dgdt = p.Rg*g*(1-g- 0.6*w)
            dwdt = p.Rw*w*(1-w-1.07*g)
        else:
            dgdt = p.Rg*g*(1-g- p.b*w*((E + g)/(p.a*E + g)))
            dwdt = p.Rw*w*(1-w- p.c*g*((p.a*E + g)/(E + g)))

        return np.array([dgdt , dwdt])

    def _dubble_u_(self, g:float,E:float)->float:
        """ w en fonction de g dans le cass par défaut"""
        return 1- self.c*g*((self.a*E + g)/(E+g))

    def _find_default_equilibria(self, E: float)->List[Equilibrium]:
        """Trouve les équilibre du sytème pour cas par défaut"""

        E1 , E2 = self.E1 , self.E2
        # les équilibres triviaux
        equilibrium = [
            Equilibrium((0,0) ),
            Equilibrium((1, 0)),
            Equilibrium((0, 1))]

        # coefficients du polynômes en g
        a1 , b1 , c1 = (1 - self.c*self.b) , (1-self.c*self.b)*self.a*E - (self.b -1 ) , 0.29*E
        delta = b1**2 -4*a1*c1
        if E1 < E < E2: # pas d'équilibres non triviaux
            return equilibrium

        elif E==E1 or E==E2: # un seul équilibre non trivial
            g = -b1/(2 * a1)
            w = self._dubble_u_(g , E)
            #if 0 < g < 1 and 0 < w1 < 1:
            equilibrium.append((g, w))
        else  :     # deux équilibres non triviaux
            g1 = (-b1 - np.sqrt(delta)) / (2 * a1)
            g2 = -b1/a1 - g1
            w1 , w2 = self._dubble_u_(g1,E) , self._dubble_u_(g2,E)

            #if 0 < g1 < 1 and 0 < w1 < 1:
            equilibrium.append((g1, w1))
            #if 0 < g2 < 1 and 0 < w2 < 1:
            equilibrium.append((g2, w2))

        return equilibrium

    def find_equilibria_val(self, *args) -> List[Equilibrium]:
        """
        Trouve les points d'équilibre du système (racines de l'équation différentielle)

        Args:
            E: Paramètre de proportionnalité avec le bétail

        Returns:
            Liste des équilibres (valeurs, type, valeurs propres)
        """
        if self.equation is self._default_equation:
            # Cas particulier: équation par défaut
            return self._find_default_equilibria(*args)
        else:
            # Cas général: recherche numérique des racines

            def eq_to_solve(state):
                """Fonction à annuler pour trouver les équilibres"""
                return self.equation(0, state, E=E)

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
                sol = fsolve(eq_to_solve, guess, full_output=True)
                if sol[2] == 1:  # La convergence a réussi
                    eq_point = sol[0]
                    # Vérification des conditions physiques
                    if all(0 <= x <= 1 for x in eq_point):
                        # Vérification que ce point n'est pas déjà trouvé
                        if not any(np.allclose(eq_point, eq.val) for eq in equilibria):
                            equilibria.append(Equilibrium(
                                val=tuple(eq_point),
                                type=None,  # Sera déterminé par classify_equilibrium
                                eigval=None
                            ))

            return equilibria


    def get_classified_equilibria(self, *agrs):
        """Classifie les points d'équilibre"""
        eqs = self.find_equilibria_val(*agrs)
        class_eq = []
        for eq in eqs:
            jac = jacobian(lambda v : self.equation(0, v, E), np.array(eq.val)) # On utilise la fonction jacobian de scipy pour calculer la matrice jacobienne
            eigenvalues = eigvals(jac)
            # Partie réelle des valeurs propres
            real_part = np.array([t.real for t in eigenvalues])
            # Les valeurs propres sont toutes de partie réelle négative
            if all(real_part<0):
                class_eq.append(Equilibrium(eq.val , 'stable', eigenvalues))
            # Les valeurs propres sont toutes de partie réelle positive
            elif all(real_part>0):
                class_eq.append(Equilibrium(eq.val , 'instable', eigenvalues))
            else:
                class_eq.append(Equilibrium(eq.val , 'selle', eigenvalues))

        return class_eq

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
    def _default_separatrix_points(self, E:float,grid_size:int=200):

        """
        Génère des points de la courbe séparatrice dans les cas par défaut

        Args:
            E: Paramètre d'exploitation ou environnemental
            grid_size: Nombre de points à générer. Defaults to 200.

        Returns:
            tuple: (w, g) où w et g sont des tableaux NumPy contenant les coordonnées des points de la courbe séparatrice. Si aucune courbe séparatrice n'existe, retourne des tableaux vides.
        """
        # On se fixe un ensemble de w entre 0 et 1
        w_grid = np.linspace(0,1,grid_size)

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

    def separatrix_points(self,):
        pass