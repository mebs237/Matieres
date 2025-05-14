import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Exemple de champ de vecteur : ici pour un systeme bidimensionnel
def f(t, u):
    g, w = u
    dgdt = g * (1 - g) - g * w
    dwdt = -w + g * w
    return [dgdt, dwdt]

# Grille pour les flèches du champ
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)
U, V = np.zeros(X.shape), np.zeros(Y.shape)

# Calcul du champ de vecteur
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u = [X[i, j], Y[i, j]]
        vec = f(0, u)
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm
        U[i, j], V[i, j] = vec

# Tracer le champ de vecteur
plt.figure(figsize=(6, 6))
plt.streamplot(X, Y, U, V, color='red', density=0.4, linewidth=0.7, arrowsize=1)

# Tracer des trajectoires pour différentes conditions initiales
initial_conditions = [
    [0.01, 0.01], [0.5, 0.9], [0.9, 0.5], [0.2, 0.8]
]
for u0 in initial_conditions:
    sol = solve_ivp(f, [0, 20], u0, t_eval=np.linspace(0, 20, 500))
    plt.plot(sol.y[0], sol.y[1], 'r-', alpha=0.8)

# Tracé d'une trajectoire spéciale en bleu
special_init = [0.05, 0.95]
sol = solve_ivp(f, [0, 20], special_init, t_eval=np.linspace(0, 20, 500))
plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=2)
plt.scatter(sol.y[0][::100], sol.y[1][::100], color='blue')

# Marquage de points fixes (à adapter selon le système)
fixed_points = [[0, 0], [1, 0], [0, 1], [1, 1]]
for pt in fixed_points:
    plt.plot(*pt, 'bo')

# Mise en forme
plt.xlabel('g')
plt.ylabel('w')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

