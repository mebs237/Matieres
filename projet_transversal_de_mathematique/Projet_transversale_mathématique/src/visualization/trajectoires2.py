import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models.Equilibres import equation

# Paramètres du système
Rg = 0.27
Rw = 0.4
E = 0.07/0.6683  # Valeur à modifier selon besoin

# Définition du système d'équations différentielles

# Temps d'intégration
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Génération des conditions initiales aléatoirement
n_ic_side = 5
initial_conditions = []

for i in range(n_ic_side):
    w0 = np.random.uniform(0, 1)
    initial_conditions.append(np.array([0,w0])) # bord gauche
    initial_conditions.append(np.array([1,w0])) # bord droit
for i in range(n_ic_side):
    g0 = np.random.normal(0, 1)
    initial_conditions.append(np.array([g0,0])) # bord haut
    initial_conditions.append(np.array([g0,1])) # bord bas


# Tracé des trajectoires
plt.figure(figsize=(10, 8))
for y0 in initial_conditions:
    sol = solve_ivp(equation, t_span, y0, t_eval=t_eval, method='RK45',args=(E,))
    plt.plot(sol.y[0], sol.y[1], 'r-', linewidth=1.5)

    # Ajout de flèches
    arrow_indices = np.linspace(0, len(sol.t)-1, 5, dtype=int)[1:-1]
    for i in arrow_indices:
        plt.arrow(sol.y[0][i], sol.y[1][i],
                 sol.y[0][i+1]-sol.y[0][i], sol.y[1][i+1]-sol.y[1][i],
                 color='red', length_includes_head=True,
                 head_width=0.02, head_length=0.03)

# Configuration du graphique
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Fraction d\'herbes vivaces (g)')
plt.ylabel('Fraction de mauvaises herbes (w)')
plt.title(f'Trajectoires du système pour E = {E}')
plt.grid(True)

# Ajout des équilibres (à calculer au préalable)
equilibria = [
    [0, 0],
    [1, 0],
    [0, 1],
    # Ajouter ici les équilibres non triviaux si nécessaire
]
for eq in equilibria:
    plt.plot(eq[0], eq[1], 'bs', markersize=10)

plt.show()