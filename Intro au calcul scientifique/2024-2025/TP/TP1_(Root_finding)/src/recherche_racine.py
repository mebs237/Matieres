import matplotlib.pyplot as plt
import numpy as np

def g(a, b):
    return b**2 * (np.exp(a) + a**3) - a**2 - 4*a

def bissection(a: float, b: float, b_value: float, n=50) -> float:
    """
    Applique l'algorithme de bissection sur la fonction g(a, b_value) dans un intervalle donné.

    Arguments :
        a : Borne inférieure de l'intervalle.
        b : Borne supérieure de l'intervalle.
        b_value : Valeur de b pour la fonction g.
        n : Nombre d'itérations maximum.

    Retour :
        float : Approximation de la racine.
    """
    while abs(b - a) > 1e-10 and n > 0:
        c = (a + b) / 2
        if g(a, b_value) * g(c, b_value) < 0:
            b = c
        else:
            a = c
        n -= 1
    return c

def continu(b: float) -> list[float]:
    """
    Trouve les valeurs de a pour lesquelles f_{a, b} est continue.
    Arguments :
        b : Valeur de b pour la fonction g.

    Retour :
        list[float] : Valeurs de a pour lesquelles f_{a, b} est continue.
    """
    if abs(b) == 0:
        return [-4, 0]

    if abs(b) == 1.18484438:
        return [0, 0.7714562069165]

    if abs(b) > 1.18484438:
        return [bissection(-4, 0, b)]

    if abs(b) < 1.18484438:
        return [bissection(-4, 0, b), bissection(0, 0.7714562069165, b), bissection(0.7714562069165, 83, b)]

# Tracer le graphe G
a_values = np.linspace(-5, 5, 2000)
b_values = np.linspace(-10, 10, 2000)
points = []

for a in a_values:
    for b in b_values:
        if abs(g(a, b)) < 1e-2:  # Vérifier si g(a, b) est proche de 0
            points.append((a, b))

# Extraire les coordonnées des points
a_coords, b_coords = zip(*points)

plt.figure(figsize=(10, 6))
plt.scatter(a_coords, b_coords, s=1)
plt.xlabel('a')
plt.ylabel('b')
plt.title('Graphe de G = {$(a, b) ∈  R^2$ | f_{a,b} est continue}')
plt.grid(True)
plt.show()