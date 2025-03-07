# Description: Module pour les fonctions continues
# Authors: PRESTONNE MEBOU
from typing import List
from numpy import expm1
from scipy.optimize import  newton

def g_b(a:float,b:float)->float:
    """
        defini dans l'énoncé du tp
    """
    return b**2*(expm1(a) + a**3)-a**2 -4*a

def dg_b(a:float,b:float)->float:
    """
        dérivée de g_b
    """
    return b**2*(expm1(a) + 3*a**2)-2*a -4

def ddg_b(a:float,b:float)->float:
    """
        dérivée seconde de g_b
    """
    return b**2*(expm1(a) + 6*a)-2

def phi(b:float)->float:
    """
        trouve le x* <0 unique racine de g_b(a) pour un b>=0
    """
    a_0 = 0
    return newton(g_b,a_0,args=(b,))


def continu(b:float)->List[float]:
    """
        ensemble des a pour lesquels f_ab est continue
    """
    a_star = phi(b)
    if dg_b(a_star,b) > 0:
        return [a_star]
    else:
        return [a_star - 1/(b**2)]