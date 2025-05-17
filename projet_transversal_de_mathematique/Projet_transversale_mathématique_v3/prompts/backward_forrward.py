import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Définir le système différentiel
def system(t, z):
    x, y = z
    dxdt = x - y
    dydt = x + y - x**2
    return [dxdt, dydt]

# Définir le système pour l'intégration à rebours
def system_backward(t, z):
    x, y = z
    dxdt = -(x - y)
    dydt = -(x + y - x**2)
    return [dxdt, dydt]

# Point de selle et point stable
saddle_point = [0, 0]
stable_point = [1, 1]

# Intervalle de temps pour l'intégration
t_span = [0, 10]
t_span_backward = [0, -10]

# Intégration en avant depuis le point de selle
sol_forward = solve_ivp(system, t_span, saddle_point, dense_output=True)

# Intégration à rebours depuis le point de selle
sol_backward = solve_ivp(system_backward, t_span_backward, saddle_point, dense_output=True)

# Grille pour le champ de phase
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Calcul du champ de vecteurs
U, V = system(0, [X, Y])

# Tracé du champ de phase
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, color='gray', alpha=0.5)

# Tracé des trajectoires
t = np.linspace(0, 10, 100)
z_forward = sol_forward.sol(t)
plt.plot(z_forward[0], z_forward[1], 'b', label='Trajectoire en avant')

t_backward = np.linspace(0, -10, 100)
z_backward = sol_backward.sol(t_backward)
plt.plot(z_backward[0], z_backward[1], 'r', label='Trajectoire en arrière')

# Points d'équilibre
plt.plot(*saddle_point, 'ko', label='Point de selle')
plt.plot(*stable_point, 'go', label='Point stable')

# Mise en forme
plt.xlabel('x')
plt.ylabel('y')
plt.title('Champ de phase avec courbes séparatrices')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
