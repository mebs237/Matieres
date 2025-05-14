import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(system, equilibria, epsilon):
    """
    Trace un champ de vecteurs autour des points d'équilibre d'un système différentiel.

    Arguments:
    system -- fonction membre de droite du système différentiel
    equilibria -- liste des points d'équilibre (coordonnées)
    epsilon -- rayon autour des équilibres
    """
    fig, ax = plt.subplots()

    for eq in equilibria:
        x_min, x_max = eq - epsilon, eq + epsilon
        y_min, y_max = eq - epsilon, eq + epsilon
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        U, V = system([X, Y])

        ax.quiver(X, Y, U, V, color='blue')
        ax.plot(eq, eq, 'ro')  # Marquer le point d'équilibre en rouge

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Champ de vecteurs autour des points d\'équilibre')
    plt.show()

# Exemple d'utilisation
def system(XY):
    x, y = XY
    dx = x - y
    dy = x + y
    return dx, dy

equilibria = [np.array([0, 0]),
              np.array([1, -1])]
epsilon = 0.5

plot_vector_field(system, equilibria, epsilon)
