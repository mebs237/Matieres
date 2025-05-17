import numpy as np
import matplotlib.pyplot as plt

E = 0.2
# Exemple de champ de vecteur : ici pour un systeme bidimensionnel
def f(t, u):
    g, w = u
    dgdt = 0.27*g * (1 - g - .6* w*(E+g)/(.31*E+g))
    dwdt = 0.4*w*(1 - w - 1.07* g*(.31*E+g)/(E+g))
    return [dgdt, dwdt]

# Grille pour les flèches du champ
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)
#U, V = np.zeros(X.shape), np.zeros(Y.shape)

U , V = f(0,[X,Y])
plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U, V)
plt.xlabel('g')
plt.ylabel('w')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
