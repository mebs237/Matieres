import matplotlib.pyplot as plt
import numpy as np

# Données d'exemple
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Création de la figure et des axes
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Premier sous-graphique
axes[0].plot(x, y1 , label='sin(x)')
axes[0].set_title('graphe 1')

# Deuxième sous-graphique
axes[1].plot(x, y2, label='cos(x)')
axes[1].set_title('graph 2')

for ax in axes:
    ax.grid(True)
    ax.legend()
    ax.set_xscale('log')

# Ajustement de l'espacement
plt.tight_layout()
# Affichage de la figure
plt.show()