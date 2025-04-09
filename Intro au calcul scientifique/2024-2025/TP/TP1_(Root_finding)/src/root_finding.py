"""
Calculate the value of a and b for which the function is continuous
"""
import math

import scipy.optimize as so
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

Function = Callable[[float], float]
def get_gb(b: float) -> Function:
    """get gb

    :param b:  parameter
    :return: the function gb
    """
    return lambda a: (b ** 2) * (math.exp(a) + a ** 3) - a ** 2 - 4 * a


def get_derivative_gb(b: float) -> Function:
    """get derivative of gb

    :param b:  parameter
    :return: the derivative of gb
    """
    return lambda a: (b ** 2) * (math.exp(a) + 3 * a ** 2) - 2 * a - 4


def get_second_derivative_gb(b: float) -> Function:
    """get second derivative of gb

    :param b:  parameter
    :return: the second derivative of gb
    """
    return lambda a: (b ** 2) * (math.exp(a) + 6 * a) - 2


def get_third_derivative_gb(b: float) -> Function:
    """get third derivative of gb

    :param b:  parameter
    :return: the third derivative of gb
    """
    return lambda a: (b ** 2) * (math.exp(a) + 6)


def h(b: float) -> float:
    """it is the function that at each b reel associate the min of the function
    gb in the positif

    :param b:  parameter
    :return: the minimum of gb in positif

    :raise ValueError: if the function is not defined
    """
    if abs(b) > 2:#pas de minimum à calculer
        raise ValueError('b there is no minimum, the function is not defined')
    if abs(b) > 0.5:
        return so.brentq(get_derivative_gb(b), 0,4)#find_root_best(get_derivative_gb(b), get_second_derivative_gb(b),0, 4)
    #on cherche la racine de la dérivée seconde pour trouver le minimum de la dérivée
    second_derivative_root = find_last_root(b, 0, get_second_derivative_gb(b),
                                            get_third_derivative_gb(b))
    #avec le minimum de la dérivée on trouve le minimum de la fonction
    return find_last_root(b, second_derivative_root, get_derivative_gb(b),
                          get_second_derivative_gb(b))


def find_last_root(b: float, x0: float, f: Function, df: Function,
                   max_iter: int = 100) -> float:
    """calculate the root of the function, the only thing we know is
    the derivative is croissant, the derivative of x0 is positive, and there is
    one root between x0 and + infinity

    :param b: parameter
    :param x0: the starting value
    :param f: the function that we want to find the root
    :param df: the derivative of f
    :param max_iter: maximum iterations
    :return: the root of the function

    :raise ValueError: if the function gb do an overflow at the root
    """
    #l'objectif est de trouver la dernière racine,
    #en partant du principe qu'il existe une racine à la dérivée et que la fonction après 
    # x0 est convexe, et que x0 est plsu grands que le minimum de la fonction
    n: int = 0
    k: float = (math.log(1 / b ** 2) - x0 / 2) * 0.5#trouver une bonne estimation du k
    #tel que la tangente de la fonction coupe l'axe x suffisament loin pour avoir une valeur positive
    fx0: float = f(x0)
    tk: float = 100
    ftk: float = f(tk)
    while ftk * fx0 > 0 and n <= max_iter:
        try:
            tk = x0 + k - f(k + x0) / df(k + x0)
            #tk l'endroit ou la tangente au point x0+k  coupe l'axe x 
        except OverflowError:
            k /= 1.5
            #si k trop grand on le diminue
        try:
            #évaluer tk pour savoir si on a un signe différent
            ftk = f(tk)
            k /= 1.08
        except OverflowError:
            #si tk est trop loin pour être évaluer faut augmenter un peu k
            k *= 1.08
        n = n + 1
    if n > max_iter:
        raise ValueError("we can not find the interval, probably "
                         "the root is to high, the limit for the exp is 710")
    return so.brentq(f, x0, tk)#find_root_best(f, df, x0, tk)


def find_b_star():
    """ find when there is only one root of gb on positif

    :return: the value of b when the root of gb and his derivative are the same
    """
    return so.brentq(lambda b: get_gb(b)(h(b)), 1, 1.5)


def continu(b: float) -> list[float]:
    """calculate the value of a for a b for which the function is continuous
    so the root of gb

    :param b: parameter
    :return: the list of the roots
    """
    if abs(b) <= 1e-151:
        return [-4, 0]
    #a1: float = find_root_best(get_gb(b), get_derivative_gb(b), -6, 0)
    a1: float = so.brentq(get_gb(b), -6, 0)
    if abs(b) >= 2:#il n'y a pas de minimum
        return [a1]
    hb: float = h(b)#le minimum dans les positif
    fhb: float = get_gb(b)(hb)#évaluer le minimum

    if fhb > 0:#le minimum est positif il n'y a pas d'autres racine
        return [a1]
    if fhb == 0:
        return [a1, hb]
    a2: float = so.brentq(get_gb(b), -0.5, hb)#find_root_best(get_gb(b), get_derivative_gb(b), -0.5, hb)
    a3: float = find_last_root(b, hb, get_gb(b), get_derivative_gb(b))
    return [a1, a2, a3]

#Question (h)
s1 = continu(1)#-1.592312824043093, 0.3293221169434157, 1.583291817156935
s2 = continu(2)#-1.0394194206751635
s3 = continu(1e-151)#-4.0, 0.0, 708.5126637027385
# Question (i)
# Tracer le graphe G

b_values = np.linspace(-10, 10, 20000)
points = []


for b in b_values:
    for a in continu(b):
        points.append((a,b))
# Extraire les coordonnées des points
a_coords, b_coords = zip(*points)
b_star = find_b_star()
plt.figure(figsize=(10, 6))
plt.axvline(x=b_star , label = 'b*', color = 'red', linestyle = '--')
plt.axvline(x=-b_star, label = "-b*", color = 'red', linestyle = '--')
plt.scatter(b_coords,a_coords,  s=1)
plt.xlabel('b')
plt.ylabel('a')
plt.title('Graphe de G = {(a, b) ∈ R^2 | f_{a,b} est continue}')
plt.grid(True)
plt.show()

# On voit que pour tous les b, on a au moins un a négatif qui rend f_{a, b} continue.
# Pour b = 0, on a effectivement uniquement -4 et 0 qui rendent f_{a, b} continue.
# Pour les b appartenant à ]-b*,b*[, on voit que 3 valeurs de a rendent f_{a, b} continue.
# Enfin, pour les b > b* ou b < -b*, on a une seule valeur de a qui rend f_{a, b} continue.