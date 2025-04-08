from numpy import ndarray , array , where , float64
from pandas import read_csv
from scipy.linalg import solve
from typing import Callable

# import du fichier de données
Data = array(read_csv("least_square25.csv") , dtype=float64)

# récupération des  (xᵢ , fᵢ)
X , f = Data[:,0] , Data[:,1]

def matrice(x:ndarray)-> dict:
    """
    étudies l'injectivité de la matrice A en fonction des xᵢ

    Return
    ---
        _ : un dictionnaire avec contenant ;
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

def solution(x:ndarray,y:ndarray)->tuple[ndarray , Callable[[ndarray],ndarray]]:
    """
        retourne l'ensemble des solutions de la procédure des moindres carrés u0 + ker(A)

        param
        ---
            x : les élongations
            y : la réponse du ressort

        Return
        ---
            _ : le tuple u0 , gen où ;
            u0 : solution particulière , minimum sur l'orthogonale de ker(A)
            gen : solutions translatée par un élément de Ker(A)
    """

    m = matrice(x)
    A , inj , l = m['A'] , m['inj'] , m['l']

    # ker(A) = {0} : unique solution u0
    if inj :
        u0 = solve(A , y)
        def gen(t):
            return u0

        return u0 , gen

    # ker(A) = R² : les solutions sont 0 + Ker(A)
    if l is None :
        u0 = array([0,0]) # orth(ker(A)) = {0}
        def gen(t1:float,t2:float)->ndarray:
            if t1<=0 or t2<0:
                raise ValueError(f" il faut t1 > 0 et t2 ≥ 0 au lieu de {t1:.3f} et {t2:.3f} ")
            return array([t1,t2])

        return  u0 , gen

    # ker(A) = span((-λ , 1)) : les solutions sont les u₀ + Ker(A)
    h0 = (1/(1+l**2))*(sum(X*f)/sum(X**2))
    u0 = h0*array([1 , l]) # car orth(ker(A)) = span((1 , λ))
    def gen(t): =  lambda t :

        return u0 + l*array([-l , 1])
    return  u0 , gen


