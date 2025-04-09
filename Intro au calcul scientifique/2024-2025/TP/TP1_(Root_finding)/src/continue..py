# Description: Module pour les fonctions continues
# Authors: PRESTONNE MEBOU
from typing import List
from numpy import exp , sqrt
from scipy.optimize import  newton

# définition des fonctions g_b , dg_b , ddg_b
def g_b(a:float,b:float)->float:
    """ fonction g_b(a) = b²( eᵃ +a³ ) -a²  -4a
    """
    return b**2*(exp(a) + a**3)-a**2 -4*a

def dg_b(a:float,b:float)->float:
    """ dérivée de g_b par rapport à a
    """
    return b**2*(exp(a) + 3*a**2)-2*a -4

def ddg_b(a:float,b:float)->float:
    """ dérivée seconde de g_b par rapport à a
    """
    return b**2*(exp(a) + 6*a)-2

def h(a:float)->float:
    """ fonction h(a) = (2a+4)/(eᵃ +3a²)
    """
    return (2*a+4)/(exp(a)+3*a**2)

def dh(a:float)->float:
    """ dérivée de h par rapport à a
    """
    return (2*exp(a)*(a**2+2*a-1))/(exp(a)+3*a**2)**2

def phi(b:float,a0:float)->float:
    """
        trouve le x* racine de g_b(a) partant de a0 , pour un b donné
    """
    return newton(g_b,a0,args=(b,))

def b_star()->float:
    """
        trouve le b* tel que pour b>b* , g_b(a) est non nulle si a>=0
        et si 0<b<b* alors g_b(a) a deux racines positives
    """

def find_b(n:int=200,info:bool=False)->float|tuple[float,int]:
    """
        trouve le b* tel que pour b>b* , g_b(a) est non nulle si a>=0
        et si 0<b<b* alors g_b(a) a deux racines positives
        b* est le supremum de l'ensemble {b ∈ ]0,√2 [ | g_b(a) a une racine positive}

        Args:
            eps : précision
            n : nombre d'itérations
            info : si True , retourne le couple (b*,a_r , n ) sinon b*
    """
    b = sqrt(2)
    arr = newton(ddg_b,0,args=(b,)) # racine de la dérivée seconde
    a_r = newton(dg_b,1.65*arr,args=(b,)) # racine de la dérivée

    while g_b(a_r,b)>0 and n>0:
        b = 0.8*b
        arr = newton(ddg_b,0,args=(b,))
        a_r = newton(dg_b,arr+1e-10,args=(b,))
        n-=1
    if info:
        return b,a_r,n
    return b

b2 , ar , ni = find_b(info=True)
print(ni)
print(ar)
print(b2)
print(g_b(ar,b2))

def continu(b:float)->List[float]:
    """
        ensemble des a bonnes racines de g_b
    """
    # Si b=0 , on a les racines sont -4 et 0
    if b==0:
        return [0,-4]
    # pour b>b2 pas de racine a positive , unique racine négative dans [-4 ,0]
    if b>b2 :
        return [phi(b,-3.6)]
    #juste deux racines positives a_1 < ar< a_2 ,et la racine négative a_3
    if 0<b<b2:
        return [phi(b,-3.6) , phi(b,0.25*ar) , phi(b,1.25*ar)]
    # pour b=b2 , on a ar comme racine positive en plus de la racine négative
    if abs(b-b2)<1e-10:
        return [phi(b,-3.6),ar]
