import numpy as np
import matplotlib.pyplot as plt
from typing import List , Tuple

def near_point(eq:Tuple[float, float], n: int, eps: float = 0.1) -> List[Tuple[float , float]]:
    """
    Trouve n points (g,w) ε-proches du point eq en
    découpant le cercle de rayon ε centré en (g,w) en n points.

    Arguments:
    eq -- coordonnées du point d'équilibre (numpy array)
    n -- nombre de points à générer sur le cercle
    eps -- rayon du cercle autour du point d'équilibre
    """
    return [(eq + eps * np.cos(2 * np.pi * i / n),
                        eq + eps * np.sin(2 * np.pi * i / n))
            for i in range(n)]

def plot_vector_field_near_equilibria(F, equilibria, eps, n_points=20)->None:
    """
    Trace les vecteurs du champ F(u) autour des points d'équilibre dans un cercle de rayon eps.

    Arguments:
    F -- fonction de champ de vecteurs, prend un tableau de coordonnées et retourne les vecteurs
    equilibria -- liste des points d'équilibre (coordonnées)
    eps -- rayon autour des équilibres
    n_points -- nombre de points à générer sur le cercle (par défaut 20)
    """
    fig = plt.subplots(figsize=(8, 6))

    for eq in equilibria:
        points = near_point(np.array(eq), n_points, eps)
        X, Y = zip(*points)
        U, V = F([np.array(X), np.array(Y)])

        plt.quiver(X, Y, U, V, color='blue', alpha=0.6)
        plt.plot(eq, eq, 'ro')  # Marquer le point d'équilibre en rouge
        circle = plt.Circle((eq, eq), eps, color='red', fill=False, linestyle='--')
        ax.add_artist(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Champ de vecteurs autour des points d\'équilibre')
    plt.show()

# Exemple d'utilisation
def F(XY):
    x, y = XY
    U = -y
    V = x
    return U, V

equilibria = [(0, 0),
              (1, -1)]
epsilon = 0.5

plot_vector_field_near_equilibria(F, equilibria, epsilon)
