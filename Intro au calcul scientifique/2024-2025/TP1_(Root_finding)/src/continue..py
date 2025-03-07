# Description: Module pour les fonctions continues
# Authors: PRESTONNE MEBOU
from typing import List
from numpy import exp , sqrt
from scipy.optimize import  newton


"""
    le but est de trouver les bonnes racines a (sielles existent ) de g_b(a) en fonction de b
    on a g_b(a) = b^2*(exp(a) + a^3)-a^2 -4a

"""

# valeur de b à partir de laquelle g_b(a) est non nulle si a>=0
b1 = sqrt(2)
# valeur de b tel si 0<=b<=b2 alors g_b(a) a deux racines positives
b2 = 1.03

def g_b(a:float,b:float)->float:
    """
        defini dans l'énoncé du tp
    """
    return b**2*(exp(a) + a**3)-a**2 -4*a

def dg_b(a:float,b:float)->float:
    """
        dérivée de g_b
    """
    return b**2*(exp(a) + 3*a**2)-2*a -4

def ddg_b(a:float,b:float)->float:
    """
        dérivée seconde de g_b
    """
    return b**2*(exp(a) + 6*a)-2

def phi(b:float,a0:float)->float:
    """
        trouve le x* <0 unique racine de g_b(a) partant de a0 , pour un b>=0
    """
    return newton(g_b,a0,args=(b,))


def continu(b:float)->List[float]:
    """
        ensemble des a bonnes racines de g_b
    """

    if b==0:
        return [0,-4]
    #pas de racine a positive
    if b>b1 or b<-b1:
        return [phi(b,-4)]
    #juste deux racines positives a_1 < ar< a_2
    if 0<b<b2:
        arr = newton(ddg_b,0,args=(b,))
        ar = newton(dg_b,arr+1e-10,args=(b,))
        return [phi(b,ar-1e-10) , phi(b,ar+1e-10)]