from numpy import exp , linspace , sqrt
import matplotlib.pyplot as plt


# Définition de la fonction fb(x, b)
fb = lambda x,b: (b**2) * (exp(x) + x**3) - x**2 - 4*x

# Création d'un tableau de valeurs x
x = linspace(-100, 100, 10000)

# Création de la figure
plt.figure(figsize=(16, 12))


# Tracer les courbes pour différentes valeurs de b
b_values = [0.0, 1e-3, 0.99, sqrt(2), 2.0, 5.0, 8.0, 50, 100]
for b in b_values:
    y = fb(x, b)
    plt.plot(x, y, label=f'b = {b}')

#Le graphique
plt.grid(True)
plt.axhline(color='k', linestyle='-', alpha=0.5)
plt.axvline(color='k', linestyle='-', alpha=0.5)
plt.xlabel('x')
plt.ylabel('$f_b(x)$')
plt.title('Graphe de $f_b(x)$ pour des $b\geq 0$')
plt.ylim(-10, 20)
plt.xlim(-8, 8)
plt.legend()

# Afficher le graphique
plt.show()