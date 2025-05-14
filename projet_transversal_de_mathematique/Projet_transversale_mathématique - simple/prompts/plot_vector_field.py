import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(F, grid_size=20):
    """
    Trace un champ de vecteurs F(u) sur une grille définie.

    Arguments:
    F -- fonction de champ de vecteurs, prend un tableau de coordonnées et retourne les vecteurs
    x_range -- tuple définissant la plage des valeurs x (xmin, xmax)
    y_range -- tuple définissant la plage des valeurs y (ymin, ymax)
    grid_size -- nombre de points dans la grille (par défaut 20)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    U, V = F([X, Y])

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Champ de vecteurs F(u)')
    plt.show()

# Exemple d'utilisation
def F(u):
    x, y = u
    dx  = -y
    dy  = x
    return np.array([dx , dy])

x_range = (-5, 5)
y_range = (-5, 5)

plot_vector_field(F, x_range, y_range)
