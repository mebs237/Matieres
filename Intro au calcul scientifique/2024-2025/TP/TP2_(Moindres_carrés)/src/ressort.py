"""
    Module pour la partie implementation du Tp
"""
from typing import Callable , Union
from numpy import ndarray , array , where , float64
from pandas import read_csv
from scipy.linalg import solve



# import du fichier de données
Data = array(read_csv("least_square25.csv") , dtype=float64)

# récupération des  (xᵢ , fᵢ)
X , f = Data[:,0] , Data[:,1]

def matrice(x:ndarray)-> dict:
    """
        calcule et étudies l'injectivité de la matrice A en fonction des xᵢ

        Returns
        ---
            un dictionnaire  contenant

            A _(ndarray)_ : matrice de la fonction ||Au-b||²
            in _(bool)_ : retourne si A est injective
            l _(float)_ : constante l = xᵢ²  lorsque A n'est pas injective
    """
    A = array([[t , t**3]  for t in x])

    # index des x_i non nuls
    index = where(x!=0)[0]

    # constante l = xᵢ²  lorsque A n'est pas injective
    l = x[index[0]] if len(index) > 0 else None

    # hypothèse (1) pour l'unicité des solutions
    inj = any(abs(x[i]) != abs(x[j]) for i in index for j in index if i != j)

    return {'A': A, 'inj': inj, 'l': l}

def _handle_unique_solution(A: ndarray, y: ndarray, contr: bool,
                          fcont: Callable[[ndarray], bool], info: bool) -> Union[ndarray, dict]:
    """
    Gère le cas où la solution est unique (ker(A) = {0}).

    Args:
        A: Matrice du système
        y: Vecteur des réponses
        contr: Active les contraintes
        fcont: Fonction de contraintes
        info: Retourne des informations supplémentaires

    Returns:
        Solution unique ou dictionnaire avec informations
    """
    u0 = solve(A.T@A, A.T@y)
    if contr and not fcont(u0):
        raise ValueError("Aucune solution ne respecte les contraintes physiques")

    if info:
        return {
            "u0": u0,
            "gen": lambda t: u0,
            "unique": True,
            "case": 0
        }
    return u0

def _handle_multiple_solutions(contr: bool, fcont: Callable[[ndarray], bool],
                             info: bool) -> Union[ndarray, dict]:
    """
    Gère le cas où il y a plusieurs solutions (ker(A) = R²).

    Args:
        contr: Active les contraintes
        fcont: Fonction de contraintes
        info: Retourne des informations supplémentaires

    Returns:
        Solution de base ou dictionnaire avec générateur
    """
    u0 = array([0, 0])  # orth(ker(A)) = {0}

    def gen(t):
        u = u0 + array([t[0], t[1]])
        if contr and not fcont(u):
            raise ValueError("Solution générée ne respecte pas les contraintes")
        return u

    if info:
        return {
            "u0": u0,
            "gen": gen,
            "unique": False,
            "case": 2
        }
    return u0

def _handle_vector_space_solution(X: ndarray, f: ndarray,
                                  l:float, contr: bool,
                                 fcont: Callable[[ndarray], bool], info: bool) -> Union[ndarray, dict]:
    """
        Gère le cas où les solutions sont dans un espace vectoriel (ker(A) = span((-λ, 1))).

        Args:
            X: Vecteur des élongations
            f: Vecteur des réponses
            l: Constante lambda
            contr: Active les contraintes
            fcont: Fonction de contraintes
            info: Retourne des informations supplémentaires

        Returns:
            Solution particulière ou dictionnaire avec générateur
    """
    h0 = (1/(1+l**2))*(sum(X*f)/sum(X**2))
    u0 = h0*array([1, l])  # orth(ker(A)) = span((1, λ))

    def gen(t):
        u = u0 + t*array([-l, 1])
        if contr and not fcont(u):
            raise ValueError("Solution générée ne respecte pas les contraintes")
        return u

    if info:
        return {
            "u0": u0,
            "gen": gen,
            "unique": False,
            "case": 1
        }
    return u0

def solution(x:ndarray,y:ndarray,contr:bool = True, fcont:Callable[[ndarray] , bool] = lambda u : u[0]>0 and u[1]>=0 , info:bool = False)->Union[ndarray , dict]:
    """
        retourne l'ensemble des solutions de la procédure des moindres carrés u0 + ker(A) sous contraintes

        Args:
            x : les élongations
            y : la réponse du ressort
            contr : si True les solutions doivent repecter les contraintes ε >0 , k >=0
            fcont : fonction de contrainte sur les solutions
            info : retourne des informations additionnelles

        Returns
        ---
            si info = False , juste une solution particulière u0

            si info = True , un dictionnaire contenant

            u₀ : solution particulière ( ε₀ , k₀ )
            gen : fonction génératrice d'autre solutions
            unique : la solution est unique ou pas
            case : taille de l'entrée du générateur

        Raise:
            ValueError: si les dimmension ne correspondent pas
            ValueError: Si aucune solution ne respecte les contraintes

    """
    # verification des dimmensions
    if x.shape != y.shape:
        raise ValueError("les dimensions de x et y doivent correspondre")
    m = matrice(x)
    A , inj , l = m['A'] , m['inj'] , m['l']

    # ker(A) = {0} : unique solution u0 = (AᵀA)⁻¹(Aᵀb)
    if inj :
        return _handle_unique_solution(A, y, contr, fcont, info)
    # ker(A) = R² : les solutions sont 0 + Ker(A)
    elif l is None :
        return _handle_multiple_solutions(contr, fcont, info)
    # ker(A) = span((-λ , 1)) : les solutions sont les u₀ + Ker(A)
    else :
        return _handle_vector_space_solution(X, f, l, contr, fcont, info)

res = solution(X,-f,info=True)
k , epsilon = res["u0"]
case = res["case"]
print("-"*20)
print(f"nous sommes dans le cas :{case} \nk={k} \nε = {epsilon}")
