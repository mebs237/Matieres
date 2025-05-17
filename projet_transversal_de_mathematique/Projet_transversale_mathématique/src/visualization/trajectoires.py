import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models.Equilibres import equation
# Paramètres du modèle
Rg = 0.27
Rw = 0.4
E = 0.1  # Valeur de E à utiliser (modifiez ici)

# Génération des conditions initiales
