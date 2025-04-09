
"""
Module pour la partie implementation du Tp
"""
from typing import Callable
from numpy import ndarray , array , where , float64
from pandas import read_csv
from scipy.linalg import solve



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


        Args:
            x : les élongations
            y : la réponse du ressort

        Returns
        ---
            u₀ : solution particulière ( ε₀ , k₀ )
            gen : fonction génératrice d'autre solutions

    """

    m = matrice(x)
    A , inj , l = m['A'] , m['inj'] , m['l']

    # ker(A) = {0} : unique solution u0
    if inj :
        u0 = solve(A , y)


        return u0 , lambda _ :u0

    # ker(A) = R² : les solutions sont 0 + Ker(A)
    if l is None :
        u0 = array([0,0]) # orth(ker(A)) = {0}


        return u0 , lambda t1 , t2 : array([t1,t2])
    else :
        # ker(A) = span((-λ , 1)) : les solutions sont les u₀ + Ker(A)
        h0 = (1/(1+l**2))*(sum(X*f)/sum(X**2))
        u0 = h0*array([1 , l]) # car orth(ker(A)) = span((1 , λ))

        return u0 , lambda t : u0 + t*array([1,l])



